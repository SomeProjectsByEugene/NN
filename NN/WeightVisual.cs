using System.Collections.Generic;
using System.Windows.Media;
using System.Windows.Shapes;

namespace NN {
    partial class NeuralNetworkVisual {
        class WeightVisual {
            static WeightVisual() {
                StrokeThickness = 5.0;
            }

            public WeightVisual(int neuronInCurrentLayer, int neuronInNextLayer, double weight) {
                var weightRGB = 255.0 / (topRandom - bottomRandom) * weight + 125;
                WeightLine = new Line {
                    Stroke = new SolidColorBrush(Color.FromRgb((byte)(255 - weightRGB), 0, (byte)weightRGB)),
                    StrokeThickness = StrokeThickness,
                    X1 = CurrentLayerNeurons[neuronInCurrentLayer].GetCenter().X,
                    X2 = NextLayerNeurons[neuronInNextLayer].GetCenter().X,
                    Y1 = CurrentLayerNeurons[neuronInCurrentLayer].GetCenter().Y,
                    Y2 = NextLayerNeurons[neuronInNextLayer].GetCenter().Y,
                };
            }

            public static List<NeuronVisual> CurrentLayerNeurons { get; set; }
            public static List<NeuronVisual> NextLayerNeurons { get; set; }

            private static double StrokeThickness { get; set; }

            public readonly Line WeightLine;
        }
    }
}
