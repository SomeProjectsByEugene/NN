using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN {
    public struct DataSet : ITraining {
        public double[] inputs { get; set; }
        public double[] targets { get; set; }
    }
}
