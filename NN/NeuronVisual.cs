using System.Windows;
using System.Windows.Media;
using System.Windows.Shapes;

namespace NN {
    partial class NeuralNetworkVisual {
        private class NeuronVisual {
            static NeuronVisual() {
                NeuronFillColor = Colors.Black;
                NeuronStrokeColor = Colors.White;
                NeuronStrokeThickness = 3.0;
            }

            public NeuronVisual(int row, int column) {
                Row = row;
                Column = column;

                NeuronEllipse = new Ellipse {
                    //Apply prepared neuron size to its width and height
                    Width = NeuronSize,
                    Height = NeuronSize,
                    //Margin depends on canvas size
                    //Here canvas is sliced on imaginary rows and columns
                    Margin = new Thickness(
                        CanvasWidth / (Layers.Length + 1) * Column - (NeuronSize / 2),
                        CanvasHeight / (Layers[Column - 1] + 1) * Row - (NeuronSize / 2),
                        0, 0
                    ),
                    //Tweaking some colors
                    Fill = new SolidColorBrush(NeuronFillColor),
                    Stroke = new SolidColorBrush(NeuronStrokeColor),
                    StrokeThickness = NeuronStrokeThickness,
                };
            }

            public static int[] Layers { get; set; }

            public static double CanvasWidth { get; set; }
            public static double CanvasHeight { get; set; }
            public static double NeuronSize { get; set; }

            private static Color NeuronFillColor { get; set; }
            private static Color NeuronStrokeColor { get; set; }

            private static double NeuronStrokeThickness { get; set; }

            public readonly Ellipse NeuronEllipse;

            public readonly int Row;
            public readonly int Column;

            /// <summary>
            /// Returns center point according to ellipse's margin and size
            /// </summary>
            /// <returns>Center point according to ellipse's margin and size</returns>
            public Point GetCenter() {
                return new Point(
                    NeuronEllipse.Margin.Left + NeuronEllipse.Width / 2,
                    NeuronEllipse.Margin.Top + NeuronEllipse.Height / 2
                );
            }
        }
    }
}
