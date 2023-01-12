using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace WeatherHistory
{
    public class LinearRegressor
    {
        private double _b0;
        private double _b1;

        public LinearRegressor()
        {
            _b0 = 0;
            _b1 = 0;
        }

        public void Fit(double[] X, double[] y)
        {
            var ssxy = X.Zip(y, (a, b) => a * b).Sum() - X.Length * X.Average() * y.Average();
            var ssxx = X.Zip(X, (a, b) => a * b).Sum() - X.Length * X.Average() * X.Average();

            _b1 = ssxy / ssxx;
            _b0 = y.Average() - _b1 * X.Average();
        }

        public double[] Predict(double[] x)
        {
            return x.Select(i => _b0 + i * _b1).ToArray();
        }
    }
}
