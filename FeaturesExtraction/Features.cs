using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using NWaves.Signals;

namespace FeaturesExtraction
{
    /// <summary>
    /// Class to manage individual features.
    /// Provides discrete signal to feature classes and processes them.
    /// After processing, data can be obtained.
    /// </summary>
    public class Features
    {
        /// <summary>
        /// KeyTonality class to provide tonality vector and key of the audio input.
        /// </summary>
        public KeyTonality KeyTonality { get; private set; }
        /// <summary>
        /// Energy class to provide information about average energy and energy deviation during audio input.
        /// </summary>
        public Energy Energy { get; private set; }
        /// <summary>
        /// Beat tracking class to provide times of beats and other information about tempo.
        /// </summary>
        public DynamicBeatTracking Rhytm { get; private set; }
        /// <summary>
        /// Harmonics class to provide information about harmony in audio input.
        /// </summary>
        public Harmonics Harmonics { get; private set; }

        DiscreteSignal Signal { get; set; }
        /// <summary>
        /// Constructs and processes signal in classes.
        /// </summary>
        /// <param name="signal"></param>
        public Features(DiscreteSignal signal)
        {
            Signal = signal;
            InitializeAndProcess();
        }

        void InitializeAndProcess()
        {
            KeyTonality = new KeyTonality(Signal);
            KeyTonality.ProcessKeyTonality();
            Energy = new Energy(Signal);
            Energy.ProcessEnergy();
            Rhytm = new DynamicBeatTracking(Signal);
            Rhytm.ProcessBeats();
            Harmonics = new Harmonics(Signal);
            Harmonics.ProcessHarmonics();
        }

        /// <summary>
        /// Returns features info.
        /// </summary>
        /// <param name="withComments">If there should be comments to be more readable.</param>
        /// <returns></returns>
        public string Data(bool withComments=false)
        {             
            StringBuilder sb = new StringBuilder();
            
            //key and tonality
            if (withComments) { sb.AppendLine("Key:"); }
            sb.AppendLine(KeyTonality.Key.ToString());
            if (withComments) { sb.AppendLine("Tonality vector:"); }
            foreach (var i in KeyTonality.TonalityVector) { sb.Append(i + " "); }
            sb.AppendLine();

            //energy
            if (withComments) { sb.AppendLine("Average energy:"); }
            sb.AppendLine(Energy.AverageEnergy.ToString());
            if (withComments) { sb.AppendLine("Energy deviation:"); }
            sb.AppendLine(Energy.StandardDeviation.ToString());

            //rhytm
            if (withComments) { sb.AppendLine("BPM:"); }
            sb.AppendLine(Rhytm.OverallTempoBPM.ToString());
            if (withComments) { sb.AppendLine("Tempo deviation (in samples):"); }
            sb.AppendLine(Rhytm.StandardInterBeatDeviation.ToString());

            //harmonics
            if (withComments) { sb.AppendLine("Average harmonic vector:"); }
            foreach(var i in Harmonics.AverageFrequencyHarmonicDistribution) { sb.Append(i + " "); }
            sb.AppendLine();
            if (withComments) { sb.AppendLine("Standard deviation of harmonic vectors:"); }
            foreach (var i in Harmonics.StandardDeviationOfFrequencyHarmonicDistribution){ sb.Append(i + " "); }
            sb.AppendLine();

            return sb.ToString();
        }

        /// <summary>
        /// Returns times of beats in seconds.
        /// </summary>
        /// <param name="withComments">If there should be comments to be more readable.</param>
        /// <returns></returns>
        public string TimeBeatsData(bool withComments = false)
        {
            StringBuilder sb = new StringBuilder();
            if (withComments) { sb.AppendLine("Time of beats in seconds:"); }
            foreach(var i in Rhytm.TimeBeats())
            {
                sb.Append(i.ToString()+" ");
            }
            sb.AppendLine();

            return sb.ToString();
        }

        /// <summary>
        /// Returns times of beats (in seconds).
        /// </summary>
        /// <returns></returns>
        public float[] TimeBeats()
        {
            return Rhytm.TimeBeats();
        }
    }


    /// <summary>
    /// FeaturesData class, mainly for serialization to JSON.
    /// </summary>
    public class FeaturesData
    {
        #region data
        //key and tonality
        /// <summary>
        /// Key of the input audio.
        /// </summary>
        public Keys Key { get; set; }
        /// <summary>
        /// Tonality vector of the input audio.
        /// </summary>
        public float[] TonalityVector { get; set; }

        //energy
        /// <summary>
        /// Average energy of the input audio.
        /// </summary>
        public float AverageEnergy { get; set; }
        /// <summary>
        /// Standard deviation of energy of the input audio.
        /// </summary>
        public float EnergyDeviation { get; set; }

        //rhytm
        /// <summary>
        /// Overall tempo in beats-per-minutes of the input audio.
        /// </summary>
        public float TempoBPM { get; set; }
        /// <summary>
        /// Standard deviation of beat intervals of the input audio.
        /// </summary>
        public float TempoDeviation { get; set; }

        //harmonics
        /// <summary>
        /// Vector of harmonics of the input audio.
        /// </summary>
        public float[] HarmonicVector { get; set; }
        /// <summary>
        /// Standard deviation of individual harmonics through time in the input audio.
        /// </summary>
        public float[] HarmonicDeviation { get; set; }
        #endregion

        /// <summary>
        /// Deserialize JSON file to FeaturesData class.
        /// </summary>
        /// <param name="jsonFile">JSON file to deserialize</param>
        /// <returns></returns>
        public static FeaturesData DeserializeJSON(string jsonFile)
        {
            using (StreamReader file = File.OpenText(jsonFile))
            {
                JsonSerializer serializer = new JsonSerializer();
                return (FeaturesData)serializer.Deserialize(file, typeof(FeaturesData));
            }
        }
    }

    /// <summary>
    /// Extension methods class.
    /// </summary>
    public static class FeaturesDataExtension
    {
        /// <summary>
        /// Returns FeaturesData object filled with values.
        /// </summary>
        /// <param name="features"></param>
        /// <returns></returns>
        public static FeaturesData GetFeaturesData(this Features features)
        {
            FeaturesData fdata = new FeaturesData()
            {
                Key = features.KeyTonality.Key,
                TonalityVector =(float[]) features.KeyTonality.TonalityVector.Clone(),

                AverageEnergy = features.Energy.AverageEnergy,
                EnergyDeviation = features.Energy.StandardDeviation,

                TempoBPM = features.Rhytm.OverallTempoBPM,
                TempoDeviation = features.Rhytm.StandardInterBeatDeviation,

                HarmonicVector = (float[])features.Harmonics.AverageFrequencyHarmonicDistribution.Clone(),
                HarmonicDeviation = (float[])features.Harmonics.StandardDeviationOfFrequencyHarmonicDistribution.Clone()
            };

            return fdata;
        }

       
    }
}
