using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN {
    class Matrix {
        public static double[,] Sum(double[,] a, double[,] b) {
            if (a.GetLength(0) != b.GetLength(0) && a.GetLength(1) != b.GetLength(1)) {
                return default(double[,]);
            }
            var result = new double[a.GetLength(0), a.GetLength(1)];
            for (int i = 0; i < result.GetLength(0); i++) {
                for (int j = 0; j < result.GetLength(1); j++) {
                    result[i, j] = a[i, j] + b[i, j];
                }
            }
            return result;
        }

        public static double[,] Subtract(double[,] a, double[,] b) {
            var result = new double[a.GetLength(0), a.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++) {
                for (int j = 0; j < a.GetLength(1); j++) {
                    result[i, j] = a[i, j] - b[i, j];
                }
            }
            return result;
        }

        public static double[,] Multiply(double[,] a, double[,] b) {
            if (a.GetLength(1) != b.GetLength(0)) {
                return default(double[,]);
            }
            var result = new double[a.GetLength(0), b.GetLength(1)];
            for (int i = 0; i < result.GetLength(0); i++) {
                for (int j = 0; j < result.GetLength(1); j++) {
                    var sum = 0.0;
                    for (int k = 0; k < b.GetLength(0); k++) {
                        sum += a[i, k] * b[k, j];
                    }
                    result[i, j] = sum;
                }
            }
            return result;
        }

        public static double[,] Transpose(double[,] a) {
            var result = new double[a.GetLength(1), a.GetLength(0)];
            for (int i = 0; i < result.GetLength(0); i++) {
                for (int j = 0; j < result.GetLength(1); j++) {
                    result[i, j] = a[j, i];
                }
            }
            return result;
        }

        public static void Map(double[,] a, Func<double, double> func) {
            for (int i = 0; i < a.GetLength(0); i++) {
                for (int j = 0; j < a.GetLength(1); j++) {
                    a[i, j] = func(a[i, j]);
                }
            }
        }

        public static double[,] AddRow(double[,] a, double[] row) {
            var result = new double[a.GetLength(0) + 1, a.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++) {
                for (int j = 0; j < a.GetLength(1); j++) {
                    result[i, j] = a[i, j];
                }
            }
            for (int i = 0; i < result.GetLength(1); i++) {
                result[result.GetLength(0) - 1, i] = row[i];
            }
            return result;
        }

        public static double[,] FromArray(double[] a) {
            var result = new double[a.Length, 1];
            for (int i = 0; i < a.Length; i++) {
                result[i, 0] = a[i];
            }
            return result;
        }

        public static double[] ToArray(double[,] a) {
            var result = new double[a.GetLength(0)];
            for (int i = 0; i < result.Length; i++) {
                result[i] = a[i, 0];
            }
            return result;
        }
    }
}
