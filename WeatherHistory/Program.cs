using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace WeatherHistory
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Problem Based on Weather History!");

            var mlcontext = new MLContext();
            var lines = System.IO.File.ReadAllLines("C:/Users/Admin/Desktop/WebPractice/MachineLearning/TemperatureProblemML/WeatherHistory/weatherHistory.csv").Skip(1).TakeWhile(t => t != null);

            List<DataPoint> itemlist = new List<DataPoint>();
            // Create a small dataset as an IEnumerable.
            foreach (var item in lines)
            {
                var values = item.Split(',');
                itemlist.Add(new DataPoint()
                {
                    Temperature = double.Parse(values[4]),
                    Humidity = double.Parse(values[5])

                });
            }

            Console.WriteLine("********Values from list before removing null*********");
            /*
            foreach (var item in itemlist)
            {
                Console.WriteLine(item.Temperature + "        " + item.Humidity);
            }
            */
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("Before---" + itemlist.Count);
            Console.ResetColor();


            // Removing null vlaues
            /*
            for (int i = 0; i < itemlist.Count; i++)
            {

                foreach (var item in itemlist.ToList())
                {
                    var itemToRemove = itemlist.FirstOrDefault(r => r.Temperature == null);

                    if (itemToRemove != null)
                        itemlist.Remove(itemToRemove);
                }
            }
            */
            Console.WriteLine("----------------------------------After Removing null values----------------------------------------");
            /*
            foreach (var item in itemlist)
            {
                Console.WriteLine(item.x + "        " + item.y);
            }
            */
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("After---" + itemlist.Count);
            Console.ResetColor();

            // Copying data to csv file
            string trainpath = @"C:/Users/Admin/Desktop/WebPractice/MachineLearning/TemperatureProblemML/WeatherHistory/weatherHistoryTwoColumns.csv";

            /* 
            StreamWriter sw = new StreamWriter(trainpath);
            CsvWriter cw = new CsvWriter(sw, CultureInfo.InvariantCulture);
            cw.WriteRecords(itemlist);
            */

            IDataView data1 = mlcontext.Data.LoadFromTextFile<DataPoint>(trainpath, hasHeader: true, separatorChar: ',');

            // NORMALIZATION

            var columnPair = new[]
            {
                new InputOutputColumnPair("Temperature"),
                new InputOutputColumnPair("Humidity")
            };
            var normalize = mlcontext.Transforms.NormalizeMeanVariance(columnPair,
                fixZero: false);
            var normalizeFixZero = mlcontext.Transforms.NormalizeMeanVariance(columnPair,
                fixZero: true);

            var normalizeTransform = normalize.Fit(data1);
            var transformedData = normalizeTransform.Transform(data1);
            var normalizeFixZeroTransform = normalizeFixZero.Fit(data1);
            var fixZeroData = normalizeFixZeroTransform.Transform(data1);
            var tempNormalized = transformedData.GetColumn<double>("Temperature");
            var humidityNormalized = transformedData.GetColumn<double>("Humidity");

            var preview = transformedData.Preview();
            foreach (var col in preview.Schema)
            {
                //Console.WriteLine(col.Index); 
                if (((col.Index % 2) == 0))
                {
                    Console.Write(col.Name + "\t");

                }
            }
            Console.WriteLine();
            for (int j = 0; j < preview.RowView.Length; j++)
            {
                for (int i = 1; i < 4; i++)
                {
                    if (i % 2 != 0)
                    {
                        Console.Write(preview.RowView[j].Values[i].Value + "\t");
                    }
                }
                Console.WriteLine();
            }
            List<DataPoint> ListOfAllColumns = new List<DataPoint>();
            //Console.WriteLine("Age\tSalary\tPurchasedList\tCountry_France\tCountry_Spain\tCountry_Germany");
            for (int j = 0; j < preview.RowView.Length; j++)
            {
                ListOfAllColumns.Add(new DataPoint()
                {
                    Temperature = ((double)preview.RowView[j].Values[1].Value),
                    Humidity = ((double)preview.RowView[j].Values[3].Value),
                });
            }
            foreach (var item in ListOfAllColumns)
            {
                Console.WriteLine(item.Temperature + " " + item.Humidity);
            }

            var dataview = mlcontext.Data.LoadFromEnumerable(ListOfAllColumns);
            var split1 = mlcontext.Data.TrainTestSplit(dataview, testFraction: 0.2);
            var trainSet = mlcontext.Data.CreateEnumerable<DataPoint>(split1.TrainSet, reuseRowObject: false);
            var testSet = mlcontext.Data.CreateEnumerable<DataPoint>(split1.TestSet, reuseRowObject: false);
            PrintPreviewRows(trainSet, testSet);

            List<double> xvalueTest = new List<double>();

            foreach (var row in trainSet)
            {
                xvalueTest.Add(row.Temperature);
            }

                List<double> yvalueTest = new List<double>();

            foreach (var row in trainSet)
            {
                yvalueTest.Add(row.Humidity);
            }

            double[] Temp_test = xvalueTest.ToArray();
            double[] Humidity_test = yvalueTest.ToArray();

            var linearRegressor = new LinearRegressor();
            var predictions =linearRegressor.Predict(Temp_test);

            double[] temp_predict = predictions.ToArray();
            List<PredictModel> PredictList = new List<PredictModel>();

            for (int j = 0; j < Temp_test.Length; j++)
            {
                PredictList.Add(new PredictModel()
                {
                    Y_Predict = temp_predict[j],
                    Y_test = Humidity_test[j]
                });
            }

            string path1 = @"C:/Users/Admin/Desktop/WebPractice/MachineLearning/TemperatureProblemML/WeatherHistory/PredictionSet.csv";
            using (StreamWriter sw = new StreamWriter(path1))
            {
                using CsvWriter cw = new CsvWriter(sw, CultureInfo.InvariantCulture);
                cw.WriteRecords(PredictList);
            }

            //calculating Accuracy
            int N = PredictList.Count;

            double[] Accuracy = { };
            List<double> AccuracyList = new List<double>();
            for (int j = 0; j < predictions.Length; j++)
            {
                AccuracyList.Add((Humidity_test[j] - temp_predict[j]));
            }

            double AccuracyValue = 0;

            Console.WriteLine("*******************Calculating Accuary by Root Mean Square!********************");
            foreach (var item in AccuracyList)
            {
                //Console.WriteLine(item.ToString());
                AccuracyValue += item;
            }

            double Square = Math.Sqrt((Math.Pow(AccuracyValue, 2)) / N);
            Console.WriteLine("Root Mean Square : " + Square);

        }
        private static void PrintPreviewRows(IEnumerable<DataPoint> trainSet,
            IEnumerable<DataPoint> testSet)

        {

            Console.WriteLine($"The data in the Train split.\n");
            foreach (var row in trainSet)
                Console.WriteLine($"{row.Temperature}, {row.Humidity}");

            Console.WriteLine($"\nThe data in the Test split.\n");
            foreach (var row in testSet)
                Console.WriteLine($"{row.Temperature}, {row.Humidity}");
        }
    }
}
