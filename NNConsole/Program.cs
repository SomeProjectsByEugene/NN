using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NN;

namespace NNConsole {
    class Program {
        private static NeuralNetwork nn;
        private static Random random;

        static void Main(string[] args) {
            random = new Random();

            var n1 = 50;

            for (int i = 1; i <= n1; i++) {
                var n2 = 10;
                var a = new double[n2];
                var b = new double[n2];
                for (int j = 0; j < n2; j++) {
                    nn = new NeuralNetwork(new[] { 2, i, 1 }, 5.0);
                    var sw = new Stopwatch();
                    sw.Start();
                    Train(10000);
                    sw.Stop();
                    a[j] = Check(10000);
                    b[j] = sw.Elapsed.TotalSeconds;
                }
                Console.WriteLine($"{i}: {a.Average()}, {b.Average()}");
            }

            //nn = new NeuralNetwork(new[] { 2, 2, 1 }, new double[] { 2.0, 3.0 });
            //Train(10000);
            //Console.WriteLine("Go!");
            //while (true) {
            //    var a = Console.ReadLine().Split(new char[] { ' ' });
            //    Console.WriteLine(nn.FeedForward(new double[] { Convert.ToDouble(a[0]), Convert.ToDouble(a[1]) })[0] + "?");
            //    //nn.Train(new ITraining[] {
            //    //    new DataSet() { inputs = new double[] { Convert.ToDouble(a[0]), Convert.ToDouble(a[1]) }, targets = new double[] { Convert.ToDouble(Console.ReadLine()) } } });
            //}
        }

        private static void Train(int n) {
            for (int i = 0; i < n; i++) {
                var i1 = (double)random.Next(2);
                var i2 = (double)random.Next(2);
                var o = ((i1 == 1 ? true : false) ^ (i2 == 1 ? true : false)) ? 1 : 0;
                nn.Train(new ITraining[] {
                    new DataSet{inputs = new double[] { i1, i2 }, targets = new double[] { o } }
                });
            }
        }

        private static double Check(int n) {
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
