using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN {
    public class NeuralNetwork {
        #region Constructors

        public NeuralNetwork(int[] layers) {
            this.layers = layers;

            //Initialize random before weights
            random = new Random();

            bottomRandom = -0.5;
            topRandom = 0.5;

            //If lerningRate is not specified, default 1.0 is used
            lerningRate = 1.0;

            InitializeWeights();
            //If biases are not specified, default zero biases are used
            InitializeBiases();
        }

        public NeuralNetwork(int[] layers, double[] biases) : this(layers) {
            InitializeBiases(biases);
        }

        public NeuralNetwork(int[] layers, double[] biases, double lerningRate) : this(layers, biases) {
            this.lerningRate = lerningRate;
        }

        public NeuralNetwork(int[] layers, double[] biases, double bottomRandom, double topRandom) : this(layers, biases) {
            NeuralNetwork.bottomRandom = bottomRandom;
            NeuralNetwork.topRandom = topRandom;
        }

        public NeuralNetwork(int[] layers, double[] biases, double lerningRate, double bottomRandom, double topRandom) : this(layers, biases) {
            NeuralNetwork.bottomRandom = bottomRandom;
            NeuralNetwork.topRandom = topRandom;
        }

        public NeuralNetwork(int[] layers, double lerningRate) : this(layers) {
            this.lerningRate = lerningRate;
        }

        public NeuralNetwork(int[] layers, double lerningRate, double bottomRandom, double topRandom) : this(layers, lerningRate) {
            NeuralNetwork.bottomRandom = bottomRandom;
            NeuralNetwork.topRandom = topRandom;
        }

        public NeuralNetwork(int[] layers, double bottomRandom, double topRandom) : this(layers) {
            NeuralNetwork.bottomRandom = bottomRandom;
            NeuralNetwork.topRandom = topRandom;
        }

        #endregion

        #region Static fields

        protected static double bottomRandom;
        protected static double topRandom;

        #endregion

        #region Fields

        //On (index) layer there're (layers[index - 1]) neurons
        protected readonly int[] layers;

        //Rows - number of neurons in current layer,
        //columns - number of neurons in previous layer
        protected List<double[,]> weights;
        protected List<double[,]> biases;

        private double lerningRate;

        private Random random;

        #endregion

        #region Activation functions

        /// <summary>
        /// Activation function - sigmoid
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private static double Sigmoid(double x) {
            return 1 / (1 + Math.Exp(-x));
        }

        #endregion

        #region Algorithms

        /// <summary>
        /// Returns outputs from neural network based on inputs
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public virtual double[] FeedForward(double[] inputs) {
            //Basic checks
            if (layers == null)
                throw new Exception();
            if (layers.Length < 1)
                throw new Exception();
            if (inputs.Length != layers[0])
                throw new Exception();

            //Pack inputs array into matrix for later feedforwarding
            var inputsMatrix = Matrix.FromArray(inputs);
            for (int i = 0; i < layers.Length - 1; i++) {
                //Add row of 1 to inputMatrix for multiplying it by bias
                inputsMatrix = Matrix.Multiply(weights[i], inputsMatrix);
                //Sum weights with biases
                inputsMatrix = Matrix.Sum(inputsMatrix, biases[i]);
                //Apply activation function, technically they're outputs
                Matrix.Map(inputsMatrix, Sigmoid);
            }
            //Return outputs from output layer
            return Matrix.ToArray(inputsMatrix);
        }

        /// <summary>
        /// Returns outputs from neural network based on inputs and list of all layers outputs
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public virtual double[] FeedForward(double[] inputs, out List<double[]> layersOutputs) {
            if (layers == null)
                throw new Exception();
            if (layers.Length < 1)
                throw new Exception();
            if (inputs.Length != layers[0])
                throw new Exception();

            layersOutputs = new List<double[]>();
            //Saving first outputs from input layer
            layersOutputs.Add(inputs);

            var inputsMatrix = Matrix.FromArray(inputs);
            for (int i = 0; i < layers.Length - 1; i++) {
                inputsMatrix = Matrix.Multiply(weights[i], inputsMatrix);
                inputsMatrix = Matrix.Sum(inputsMatrix, biases[i]);
                Matrix.Map(inputsMatrix, Sigmoid);
                //Saving new outputs
                layersOutputs.Add(Matrix.ToArray(inputsMatrix));
            }
            return Matrix.ToArray(inputsMatrix);
        }

        /// <summary>
        /// Trains network using given datasets
        /// </summary>
        /// <param name="dataSets"></param>
        public virtual void Train(IEnumerable<ITraining> dataSets) {
            foreach (var dataSet in dataSets) {
                //Results of currently trained network
                var outputs = FeedForward(dataSet.inputs, out var layersOutputs);

                ////Get errors matrices
                //var errors = GetErrors(dataSet.targets, outputs);

                ////Adjusting weights
                //AdjustWeights(errors, layersOutputs);
                //AdjustBiases(errors, layersOutputs);

                double[,] errors = null;
                for (int i = layers.Length - 2; i >= 0; i--) {
                    if (i == layers.Length - 2) {
                        errors = Matrix.Subtract(Matrix.FromArray(dataSet.targets), Matrix.FromArray(outputs));
                    } else {
                        errors = Matrix.Multiply(Matrix.Transpose(weights[i + 1]), errors);
                    }
                    var gradients = Matrix.FromArray(layersOutputs[i + 1]);
                    //Matrix.Map(gradients, (x) => x * (x - 1) * );
                    for (int j = 0; j < gradients.GetLength(0); j++) {
                        gradients[j, 0] *= errors[j, 0]; 
                    }
                    biases[i] = gradients;
                    //gradients = Matrix.Multiply(Matrix.Transpose(gradients), errors);
                    gradients = Matrix.Multiply(gradients, Matrix.Transpose(Matrix.FromArray(layersOutputs[i])));
                    Matrix.Map(gradients, (x) => x * lerningRate);
                    weights[i] = Matrix.Sum(weights[i], gradients);
                }
            }
        }

        /// <summary>
        /// Returns the collection of errors matrices for each layer except input one
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="outputs"></param>
        /// <returns></returns>
        private double[][,] GetErrors(double[] targets, double[] outputs) {
            //Initialize errors array of matrices
            var errors = new double[layers.Length - 1][,];
            //Calculate errors of output layer
            errors[layers.Length - 2] =
                Matrix.Subtract(Matrix.FromArray(targets), Matrix.FromArray(outputs));
            //Calculate errors of other layers, except input one
            for (int i = layers.Length - 3; i >= 0; i--) {
                errors[i] =
                    Matrix.Multiply(Matrix.Transpose(weights[i + 1]), errors[i + 1]);
            }
            return errors;
        }
        
        /// <summary>
        /// Adjusts weights
        /// </summary>
        /// <param name="layersOutputs"></param>
        private void AdjustWeights(double[][,] errors, List<double[]> layersOutputs) {
            //Adjusting weights according to errors and layers outputs
            //From right to left
            for (int i = layers.Length - 2; i >= 0; i--) {
                for (int j = 0; j < weights[i].GetLength(0); j++) {
                    for (int k = 0; k < weights[i].GetLength(1) - 1; k++) {
                        //DeltaWeight - an amount that you should to subtract from weight in order to adjust it
                        //FOR ME, using otputs in ( outputs*(1-outputs) ) instead of inputs
                        var deltaWeight = -errors[i][j, 0] * layersOutputs[i + 1][j] * (1 - layersOutputs[i + 1][j]) * layersOutputs[i][k];
                        //Multiplying by lerning rate to controll speed of learning
                        deltaWeight *= lerningRate;
                        //Adjusting weight
                        weights[i][j, k] -= deltaWeight;
                    }
                }
            }
        }

        /// <summary>
        /// Adjusts biases
        /// </summary>
        /// <param name="layersOutputs"></param>
        private void AdjustBiases(double[][,] errors, List<double[]> layersOutputs) {
            //Adjusting weights according to errors and layers outputs
            //From right to left
            for (int i = layers.Length - 2; i >= 0; i--) {
                for (int j = 0; j < weights[i].GetLength(0); j++) {
                    //DeltaBias - an amount that you should to subtract from bias in order to adjust it
                    var deltaBias = -errors[i][j, 0] * layersOutputs[i + 1][j] * (1 - layersOutputs[i + 1][j]);
                    //Multiplying by lerning rate to controll speed of learning
                    deltaBias *= lerningRate;
                    //Adjusting weight
                    weights[i][j, weights[i].GetLength(1) - 1] -= deltaBias;
                }
            }
        }

        #endregion

        #region Components initialization

        /// <summary>
        /// Initialize weights using random and biases
        /// </summary>
        private void InitializeWeights() {
            weights = new List<double[,]>();
            for (int i = 1; i < layers.Length; i++) {
                //Between each two layers there's weights matrix
                var weightsBetweenLayers = new double[layers[i], layers[i - 1]];
                //Randomizing weights
                for (int j = 0; j < weightsBetweenLayers.GetLength(0); j++) {
                    for (int k = 0; k < weightsBetweenLayers.GetLength(1); k++) {
                        weightsBetweenLayers[j, k] = random.NextDouble() * (topRandom - bottomRandom) + bottomRandom;
                    }
                }
                weights.Add(weightsBetweenLayers);
            }
        }

        /// <summary>
        /// Initialize biases with zeros
        /// </summary>
        /// <param name="biases"></param>
        private void InitializeBiases() {
            this.biases = new List<double[,]>();
            for (int i = 1; i < layers.Length; i++) {
                var currentBias = new double[layers[i]];
                for (int j = 0; j < currentBias.Length; j++) {
                    currentBias[j] = 0.0;
                }
                this.biases.Add(Matrix.FromArray(currentBias));
            }
        }

        /// <summary>
        /// Initialize biases and validate their length
        /// </summary>
        /// <param name="biases"></param>
        private void InitializeBiases(double[] biases) {
            if (biases.Length != layers.Length - 1) {
                throw new Exception();
            }
            this.biases = new List<double[,]>();
            for (int i = 0; i < biases.Length; i++) {
                var currentBias = new double[layers[i + 1]];
                for (int j = 0; j < currentBias.Length; j++) {
                    currentBias[j] = biases[i];
                }
                this.biases.Add(Matrix.FromArray(currentBias));
            }
        }

        ///// <summary>
        ///// Initialize biases with random values
        ///// </summary>
        ///// <param name="biases"></param>
        //private void InitializeBiases() {
        //    for (int i = 0; i < layers.Length - 1; i++) {
        //        for (int j = 0; j < weights[i].GetLength(0); j++) {
        //            weights[i][j, weights[i].GetLength(1) - 1] = random.NextDouble() * (topRandom - bottomRandom) + bottomRandom;
        //        }
        //    }
        //}

        ///// <summary>
        ///// Initialize biases and validate their length
        ///// </summary>
        ///// <param name="biases"></param>
        //private void InitializeBiases(double[] biases) {
        //    if (biases.Length != layers.Length - 1) {
        //        throw new Exception();
        //    }
        //    for (int i = 0; i < layers.Length - 1; i++) {
        //        for (int j = 0; j < weights[i].GetLength(0); j++) {
        //            weights[i][j, weights[i].GetLength(1) - 1] = biases[i];
        //        }
        //    }
        //}

        #endregion
    }
}
