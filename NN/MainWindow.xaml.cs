using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace NN {
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window {
        public MainWindow() {
            InitializeComponent();

            timer.Tick += TimerOnTick;
            Loaded += OnLoaded;
            MainCanvas.MouseWheel += OnMainCanvasMouseWheel;
        }

        private void TimerOnTick(object sender, EventArgs e) {
            Train(1);
            //nn.Draw();
        }

        private const double SCALE_COEFFICIENT = 1.05;

        private NeuralNetworkVisual nn;
        private Random random = new Random();
        private DispatcherTimer timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(1) };

        private void OnLoaded(object sender, RoutedEventArgs e) {
            nn = new NeuralNetworkVisual(MainCanvas, new[] { 2, 2, 1 });
            timer.Start();
            //Train(1);
        }

        private void OnMainCanvasMouseWheel(object sender, MouseWheelEventArgs e) {
            var coordinates = e.GetPosition(MainCanvas);
            MainCanvasScaleTransform.CenterX = coordinates.X;
            MainCanvasScaleTransform.CenterY = coordinates.Y;
            
            if (e.Delta > 0) {
                MainCanvasScaleTransform.ScaleX *= SCALE_COEFFICIENT;
                MainCanvasScaleTransform.ScaleY *= SCALE_COEFFICIENT;
            } else {
                MainCanvasScaleTransform.ScaleX /= SCALE_COEFFICIENT;
                MainCanvasScaleTransform.ScaleY /= SCALE_COEFFICIENT;
            }
        }

        private void Train(int n) {
            for (int i = 0; i < n; i++) {
                var i1 = (double)random.Next(2);
                var i2 = (double)random.Next(2);
                var o = ((i1 == 1 ? true : false) ^ (i2 == 1 ? true : false)) ? 1 : 0;
                nn.Train(new ITraining[] {
                    new DataSet{inputs = new double[] { i1, i2 }, targets = new double[] { o } }
                });
            }
        }

        private double Check(int n) {
            //var correct = 0;
            //var incorrect = 0;
            var a = new double[n];
            for (int i = 0; i < n; i++) {
                var i1 = (double)random.Next(2);
                var i2 = (double)random.Next(2);
                var o = ((i1 == 1 ? true : false) ^ (i2 == 1 ? true : false)) ? 1 : 0;
                a[i] = Math.Abs(nn.FeedForward(new double[] { i1, i2 })[0] - o);
            }
            return a.Average();
        }
    }
}
