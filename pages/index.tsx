import * as React from 'react';
import * as Chart from 'chart.js';
import {
  AdadeltaOptimizer,
  ConstantInitializer,
  Graph,
  GraphRunner,
  InGPUMemoryShuffledInputProviderBuilder,
  NDArray,
  NDArrayMathGPU,
  RandomNormalInitializer,
  Session,
  xhr_dataset,
  XhrDataset,
} from 'deeplearn';

interface State {
  totalBatchesTrained: number;
  trainExamplesPerSec: number;
  totalTimeSec: number;
  cost: number;
}

const TRAIN_INFER_RATIO = 0.9;
const SIZE = 28;
const INPUT_VECTOR_LENGTH = SIZE * SIZE;
const INITIAL_LEARNING_RATE = 0.01;
const BATCH_SIZE = 32;
const INFERENCE_EXAMPLE_INTERVAL_MS = 10000;
const INFERENCE_EXAMPLE_COUNT = 5;

export default class extends React.Component<void, State> {
  private costChart: Chart;
  private inputContexts: CanvasRenderingContext2D[] = [];
  private outputContexts: CanvasRenderingContext2D[] = [];

  constructor(props: void) {
    super(props);
    this.state = {
      totalBatchesTrained: 0,
      trainExamplesPerSec: 0,
      totalTimeSec: 0,
      cost: 0
    }
  }

  render(): JSX.Element {
    const { totalBatchesTrained, trainExamplesPerSec, totalTimeSec, cost } = this.state;
    const inferenceCanvases: JSX.Element[] = [];
    for(let i = 0; i < INFERENCE_EXAMPLE_COUNT; i++) {
      inferenceCanvases.push(<div key={i}>
        <canvas ref={canvas => {
          if (this.inputContexts[i]) {
            return;
          }
          canvas!.width = SIZE;
          canvas!.height = SIZE;
          this.inputContexts[i] = canvas!.getContext('2d')!;
        }}/>
        <canvas ref={canvas => {
          if (this.outputContexts[i]) {
            return;
          }
          canvas!.width = SIZE;
          canvas!.height = SIZE;
          this.outputContexts[i] = canvas!.getContext('2d')!;
        }}/>
      </div>)
    }
    return (
      <div>
        <p>Trained batches: {totalBatchesTrained}</p>
        <p>Train Examples / sec: {trainExamplesPerSec}</p>
        <p>Total time in sec: {totalTimeSec}</p>
        <p>Cost: {cost}</p>
        <canvas ref={canvas => !this.costChart && (this.costChart = this.createLineChart(canvas!, 'cost', 0))} />
        {inferenceCanvases}
      </div>
    );
  }

  componentDidMount(): void {
    xhr_dataset.getXhrDatasetConfig('static/dataset.config.json')
      .then(({ mnist }) => {
        const dataset = new XhrDataset(mnist);
        dataset.fetchData().then(() => {
          const graph = new Graph();
          const rni = new RandomNormalInitializer(0, 0.1);
          const ci = new ConstantInitializer(0.001);
          const x = graph.placeholder('x', [INPUT_VECTOR_LENGTH]);
          const math = new NDArrayMathGPU();

          // 1st encoding phase
          const WEncode1 = graph.variable('WEncode1', rni.initialize([INPUT_VECTOR_LENGTH, 256], 0, 0));
          const bEncode1 = graph.variable('bEncode1', ci.initialize([256], 0, 0));
          const hEncode1 = graph.sigmoid(graph.add(graph.matmul(x, WEncode1), bEncode1));

          // 2nd encoding phase
          const WEncode2 = graph.variable('WEncode2', rni.initialize([256, 128], 0, 0));
          const bEncode2 = graph.variable('bEncode2', ci.initialize([128], 0, 0));
          const hEncode2 = graph.sigmoid(graph.add(graph.matmul(hEncode1, WEncode2), bEncode2));

          // 1st decoding phase
          const WDecode1 = graph.variable('WDecode1', rni.initialize([128, 256], 0, 0));
          const bDecode1 = graph.variable('bDecode1', ci.initialize([256], 0, 0));
          const hDecode1 = graph.sigmoid(graph.add(graph.matmul(hEncode2, WDecode1), bDecode1));

          // 2nd decoding phase
          const WDecode2 = graph.variable('WDecode2', rni.initialize([256, INPUT_VECTOR_LENGTH], 0, 0));
          const bDecode2 = graph.variable('bDecode2', ci.initialize([INPUT_VECTOR_LENGTH], 0, 0));
          const hDecode2 = graph.sigmoid(graph.add(graph.matmul(hDecode1, WDecode2), bDecode2));
          const costTensor = graph.meanSquaredCost(x, hDecode2);

          // create GraphRunner
          const session = new Session(graph, math);
          const graphRunner: GraphRunner = new GraphRunner(math, session, {
            batchesTrainedCallback: totalBatchesTrained => this.setState({ totalBatchesTrained }),
            avgCostCallback: avgCost => this.updateChart(graphRunner.getTotalBatchesTrained(), avgCost.get()) || this.setState({ cost: avgCost.get() }),
            trainExamplesPerSecCallback: trainExamplesPerSec => this.setState({ trainExamplesPerSec }),
            totalTimeCallback: totalTimeSec => this.setState({ totalTimeSec }),
            inferenceExamplesCallback: (inputFeeds, inferenceOutput) => {
              const input = inputFeeds.map(([input]) => input.data as NDArray);
              const unnormalizedInput = dataset.unnormalizeExamples(input, 0);
              const unnormalizedOutput = dataset.unnormalizeExamples(inferenceOutput, 0);
              unnormalizedInput.forEach((input, i) => {
                this.putImageData(this.inputContexts[i], input);
                this.putImageData(this.outputContexts[i], unnormalizedOutput[i]);
              })
            }
          });

          // prepare dataset
          dataset.normalizeWithinBounds(0, 0, 1);
          const [rawImages, labels] = dataset.getData()!;
          const images: NDArray[] = [];
          for (const rawImage of rawImages) {
            images.push(rawImage.reshape([INPUT_VECTOR_LENGTH]));
          }
          const length = images.length, end = Math.floor(TRAIN_INFER_RATIO * length);

          // train
          const [trainInputProvider] = new InGPUMemoryShuffledInputProviderBuilder([images.slice(0, end), labels.slice(0, end)]).getInputProviders();
          graphRunner.train(
            costTensor,
            [{ tensor: x, data: trainInputProvider }],
            BATCH_SIZE,
            new AdadeltaOptimizer(INITIAL_LEARNING_RATE, 0.01)
          );

          // infer
          const [inferInputProvider] = new InGPUMemoryShuffledInputProviderBuilder([images.slice(end), labels.slice(end)]).getInputProviders();
          graphRunner.infer(
            hDecode2,
            [{ tensor: x, data: inferInputProvider}],
            INFERENCE_EXAMPLE_INTERVAL_MS,
            INFERENCE_EXAMPLE_COUNT
          );
        })

      });
  }

  private putImageData(context: CanvasRenderingContext2D, data: NDArray): void {
    const buffer = new Uint8ClampedArray(INPUT_VECTOR_LENGTH * 4);
    const dataValues = data.getValues();
    for (let i = 0; i < INPUT_VECTOR_LENGTH; i++) {
      const pos = i * 4;
      const value = Math.round(Math.max(Math.min(dataValues[i], 255), 0));
      buffer[pos] = value;
      buffer[pos + 1] = value;
      buffer[pos + 2] = value;
      buffer[pos + 3] = 255;
    }
    const imageData = context.createImageData(SIZE, SIZE);
    imageData.data.set(buffer);
    context.putImageData(imageData, 0, 0);
  }

  private createLineChart(canvas: HTMLCanvasElement, label: string, min?: number, max?: number): Chart {
    return new Chart(canvas, {
      type: 'line',
      data: {
        datasets: [{
          data: [],
          fill: false,
          pointRadius: 0,
          borderColor: 'rgba(75,192,192,1)',
          borderWidth: 1,
          lineTension: 0,
          pointHitRadius: 8,
          label,
        }]
      },
      options: {
        responsive: false,
        scales: {
          xAxes: [{type: 'linear', position: 'bottom'}],
          yAxes: [{
            ticks: {
              max,
              min,
            }
          }]
        }
      }
    });
  }

  private updateChart(x: number, y: number): void {
    (this.costChart.data!.datasets![0].data as { x: number; y: number; }[]).push({ x, y });
    this.costChart.update();
  }
}
