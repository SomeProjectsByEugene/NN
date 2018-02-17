using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace NN {
    partial class NeuralNetworkVisual : NeuralNetwork {
        public NeuralNetworkVisual(Canvas canvas, int[] layers) : base(layers) {
            this.canvas = canvas;
            this.canvas.SizeChanged += OnCanvasSizeChanged;

            neurons = new List<NeuronVisual>();

            NeuronVisual.Layers = layers;

            Draw();
        }

        public NeuralNetworkVisual(Canvas canvas, int[] layers, double[] biases) : base(layers, biases) {
        }

        public NeuralNetworkVisual(Canvas canvas, int[] layers, double lerningRate) : base(layers, lerningRate) {
        }

        public NeuralNetworkVisual(Canvas canvas, int[] layers, double[] biases, double lerningRate) : base(layers, biases, lerningRate) {
        }

        public NeuralNetworkVisual(Canvas canvas, int[] layers, double bottomRandom, double topRandom) : base(layers, bottomRandom, topRandom) {
        }

        public NeuralNetworkVisual(Canvas canvas, int[] layers, double[] biases, double bottomRandom, double topRandom) : base(layers, biases, bottomRandom, topRandom) {
        }

        public NeuralNetworkVisual(Canvas canvas, int[] layers, double[] biases, double lerningRate, double bottomRandom, double topRandom) : base(layers, biases, lerningRate, bottomRandom, topRandom) {
        }

        private const double NEURON_SIZE_COEFFICIENT = 15.0;

        private readonly Canvas canvas;

        private readonly List<NeuronVisual> neurons;

        /// <summary>
        /// Draws network on canvas
        /// </summary>
        public void Draw() {
            //Delete all items from canvas
            //For example, after redrawing network on this canvas
            canvas.Children.Clear();
            neurons.Clear();
            DrawNeurons();
            DrawWeights();
        }

        public override double[] FeedForward(double[] inputs) {
            for (int i = 0; i < inputs.Length; i++) {
                var b = neurons.Where(n => n.Row == i + 1 && n.Column == 1).FirstOrDefault();
                var c = b.GetCenter();
                var textInput = new TextBlock {
                    Text = inputs[i].ToString(),
                    Margin = new Thickness(c.X - 3, c.Y - 6, 0, 0),
                    FontSize = 10,
                    Foreground = new SolidColorBrush(Colors.Red),
                };
                Canvas.SetZIndex(textInput, 2);
                canvas.Children.Add(textInput);
            }
            
            return base.FeedForward(inputs);
        }

        public override void Train(IEnumerable<ITraining> dataSets) {
            base.Train(dataSets);

        }

        private void DisplayError() {
            var text = new TextBlock {
                Margin = new Thickness(0, 0, 0, 0),
                Foreground = new SolidColorBrush(Colors.White),
                FontSize = 14
            };

        }

        //private double GetError(int checksCount) {
        //    var errors = new double[checksCount];
        //    //for (int i = 0; i < checksCount; i++) {
        //    //    errors[i] = 
        //    //}
        //}

        /// <summary>
        /// Draws neurons on canvas
        /// </summary>
        private void DrawNeurons() {
            NeuronVisual.CanvasWidth = canvas.ActualWidth;
            NeuronVisual.CanvasHeight = canvas.ActualHeight;
            //Neuron size depends on canvas size
            //More bigger canvas - more bigger ellipse of neuron
            NeuronVisual.NeuronSize =
                //Actually it depends on smallest dimention: either width or height
                (canvas.ActualWidth > canvas.ActualHeight ?
                canvas.ActualHeight : canvas.ActualWidth) / NEURON_SIZE_COEFFICIENT;

            //Creating neurons for each imaginary row and column
            for (int i = 1; i <= layers.Length; i++) {
                for (int j = 1; j <= layers[i - 1]; j++) {
                    var neuron = new NeuronVisual(j, i);
                    neurons.Add(neuron);
                    //Adding new neuron ellipse to canvas
                    Canvas.SetZIndex(neuron.NeuronEllipse, 1);
                    canvas.Children.Add(neuron.NeuronEllipse);
                }
            }
        }

        /// <summary>
        /// Draws weights on canvas
        /// </summary>
        private void DrawWeights() {
            for (int i = 1; i < layers.Length; i++) {
                //Set new info about two layers, between which we're situated
                WeightVisual.CurrentLayerNeurons = neurons.Where(
                    n => n.Column == i).ToList();
                WeightVisual.NextLayerNeurons = neurons.Where(
                    n => n.Column == i + 1).ToList();

                //Loop through each neuron in current...
                for (int j = 0; j < WeightVisual.CurrentLayerNeurons.Count; j++) {
                    //... and next layer
                    for (int k = 0; k < WeightVisual.NextLayerNeurons.Count; k++) {
                        var weight = new WeightVisual(j, k, weights[i - 1][k, j]);
                        Canvas.SetZIndex(weight.WeightLine, 0);
                        canvas.Children.Add(weight.WeightLine);
                    }
                }
            } 
        }

        private void OnCanvasSizeChanged(object sender, SizeChangedEventArgs e) {
            Draw();
        }
    }
}
