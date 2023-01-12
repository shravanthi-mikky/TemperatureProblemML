using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace WeatherHistory
{
    public class DataPoint
    {
        [LoadColumn(0)]
        public double Temperature { get; set; }

        [LoadColumn(1)]
        public double Humidity { get; set; }
    }

    public class PredictModel
    {
        [LoadColumn(0)]
        public double Y_Predict { get; set; }
        [LoadColumn(1)]
        public double Y_test { get; set; }
    }
}
