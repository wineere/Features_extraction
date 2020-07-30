using NWaves.Signals;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FeaturesExtraction
{
    /// <summary>
    /// Class that computes harmonic distribution in provided audio signal.
    /// </summary>
    public class Harmonics
    {
        DiscreteSignal Signal { get; }
        /// <summary>
        /// Indicates if Harmonics class is processed.
        /// </summary>
        public bool Processed { get; private set; }

        /// <summary>
        /// List contains float arrays which contains absolute values of complex numbers produced by Short-time fourier transform.
        /// </summary>
        List<float[]> StftAbsoluteValue { get; set; }
        /// <summary>
        /// List contains float arrays which describe harmonic distribution of frequencies in corresponding frame of STFT.
        /// </summary>
        List<float[]> HarmonicDistribution { get; set; }

        /// <summary>
        /// Maximum number of harmonics considered.
        /// </summary>
        public int NumberOfHarmonics { get; private set; } = 20; //random value, couldn't find any referential value in paper
        /// <summary>
        /// Average frequency harmonic distribution from individuall frames of STFT. 
        /// </summary>
        public float[] AverageFrequencyHarmonicDistribution { get; private set; }
        /// <summary>
        /// Standard deviation of harmonic distribution from individuall frames of STFT.
        /// </summary>
        public float[] StandardDeviationOfFrequencyHarmonicDistribution { get; private set; }


        int STFTWindowsSize { get; set; } = 1024;   //default value of stft implementation, in paper no specified value.
        int STFTHopSize { get; set; } = 256;        //default value of stft implementation, in paper no specified value.


        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="signal"></param>
        public Harmonics(DiscreteSignal signal)
        {
            Signal = signal;
        }

       
        /// <summary>
        /// Computes harmonics.
        /// </summary>
        public void ProcessHarmonics()
        {
            if (Processed)
            {
                return;
            }

            ComputeStftAbsoluteValue();
            ComputeHarmonicDistribution();
            ComputeAverageAndDeviation();
            Processed = true;
        }

        /// <summary>
        /// Computes STFT and then absolute values of computed complex numbers and stores it to StftAbsoluteValues.
        /// </summary>
        void ComputeStftAbsoluteValue()
        {
            NWaves.Transforms.Stft stft = new NWaves.Transforms.Stft(STFTWindowsSize, STFTHopSize);
            var stftOutput = stft.Direct(Signal);

            var list = new List<float[]>();

            foreach(var tuple in stftOutput)
            {
                var re = tuple.Item1;
                var im = tuple.Item2;
                float[] abs = new float[re.Length];

                for (int i = 0; i < re.Length; i++)
                {
                    abs[i] = (float)Math.Sqrt(re[i] * re[i] + im[i] * im[i]);
                }

                list.Add(abs);
            }

            StftAbsoluteValue = list;
        }
        /// <summary>
        /// From StftAbsoluteValues computes harmonic distribution in each frame and stores it to HarmonicDistribution.
        /// </summary>
        void ComputeHarmonicDistribution()
        {
            var list = new List<float[]>();
            //for each frame we compute harmonics distribution
            foreach (var frame in StftAbsoluteValue)
            {
                float[] frameHarmonics = new float[frame.Length/NumberOfHarmonics];
                //for each fundamental frequency we compute distribution using formula (4) from paper 
                for (int f = 1; f < frameHarmonics.Length; f++)
                {
                    //sum in the formula (4)
                    for (int k = 1; k < NumberOfHarmonics+1; k++)
                    {
                        frameHarmonics[f] += Math.Min(frame[f], frame[k * f]);
                    }
                }
                list.Add(frameHarmonics);
            }
            HarmonicDistribution = list;
        }
        /// <summary>
        /// From HarmonicDistribution computes Average and Standard deviation.
        /// </summary>
        void ComputeAverageAndDeviation()
        {
            var tuple = HarmonicsPostprocessing.AverageValueAndStandardDeviation(HarmonicDistribution);
            AverageFrequencyHarmonicDistribution = tuple.Item1;
            StandardDeviationOfFrequencyHarmonicDistribution = tuple.Item2;
        }
    }

    /// <summary>
    /// Auxiliary class to provide computational tools.
    /// </summary>
    public static class HarmonicsPostprocessing
    {
        /// <summary>
        /// From list of vectors computes average vector and standard deviation vector.
        /// </summary>
        /// <param name="values">List of vectors of the same size.</param>
        /// <returns></returns>
        public static (float[],float[]) AverageValueAndStandardDeviation(List<float[]>values)
        {
            var average = ComputeAverage(values);
            var deviation = ComputeStandardDeviation(average, values);
            return (average, deviation);
        }

        /// <summary>
        /// Computes average vector from provided vectors.
        /// </summary>
        /// <param name="values">List of vectors of the same size.</param>
        /// <returns></returns>
        static float[] ComputeAverage(List<float[]>values)
        {
            var avg = new float[values[0].Length];
            //sum each frame to avg
            foreach (var frame in values)
            {
                for (int i = 0; i < avg.Length; i++)
                {
                    avg[i] += frame[i];
                }
            }

            for (int i = 0; i < avg.Length; i++)
            {
                avg[i] = avg[i] / values.Count;
            }

            return avg;
        }
        /// <summary>
        /// Computes standard deviation of vectors from average vector.
        /// </summary>
        /// <param name="average">Average vector.</param>
        /// <param name="values">List of vectors of the same size.</param>
        /// <returns></returns>
        static float[] ComputeStandardDeviation(float[]average, List<float[]>values)
        {
            var dev = new float[average.Length];
            //for each frame we add squared difference
            foreach (var frame in values)
            {
                for (int i = 0; i < dev.Length; i++)
                {
                    dev[i] = (average[i] - frame[i]) * (average[i] - frame[i]);
                }
            }

            //now divide by number of samples minus 1 and do square root
            for (int i = 0; i < dev.Length; i++)
            {
                dev[i] = dev[i] / (values.Count - 1);
                dev[i] = (float)Math.Sqrt(dev[i]);
            }

            return dev;
        }
    }

}
