using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using NWaves.Signals;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Newtonsoft.Json;
using System.IO;

namespace FeaturesExtraction
{
    /// <summary>
    /// Static class to provide final methods to be called on input file.
    /// </summary>
    public static class ProgramOutput
    {
        /// <summary>
        /// Computes beat times in inFile, add clap sound to computed times and saves to file.
        /// </summary>
        /// <param name="inputFile">MP3 file</param>
        public static void AddClapsOnRhytm(string inputFile)
        {
            string outFileName = System.IO.Path.GetFileNameWithoutExtension(inputFile) + "_with_clap.wav";


            //https://sampleswap.org/filebrowser-new.php?d=DRUMS+%28SINGLE+HITS%29%2FClaps%2F
            string clapFile = "data_files/clap.mp3";

            DiscreteSignal inputSignal = Tools.ReadSignalFromMp3(inputFile);
            DynamicBeatTracking rhytm = new DynamicBeatTracking(inputSignal);
            rhytm.ProcessBeats();

            var clapOnBeats = Tools.ClappingToBeats(rhytm.TimeBeats(), clapFile);
            ISampleProvider audio = new Mp3FileReader(inputFile).ToSampleProvider().ToMono();

            MixingSampleProvider mix = new MixingSampleProvider(new ISampleProvider[] { audio, clapOnBeats });
                        
            WaveFileWriter.CreateWaveFile16(outFileName, mix);            
        }

        /// <summary>
        /// Extracts features from inputFile and saves them to text file.
        /// </summary>
        /// <param name="inputFile">MP3 file</param>
        /// <param name="comments">If comments should be included in outfile.</param>
        public static void ExtractFeatures(string inputFile, bool comments)
        {
            string outFile = System.IO.Path.GetFileNameWithoutExtension(inputFile) + "_features_data.txt";

            DiscreteSignal inputSignal = Tools.ReadSignalFromMp3(inputFile);
            Features f = new Features(inputSignal);
            System.IO.File.WriteAllText(outFile, f.Data(comments));
        }

        /// <summary>
        /// Extracts features from input file and saves them to JSON file.
        /// </summary>
        /// <param name="inputFile"></param>
        /// <param name="outFile"></param>
        public static void ExtractFeaturesAndSaveToJSON(string inputFile, string outFile = null)
        {
            if (outFile == null)
            {
                outFile = System.IO.Path.GetFileNameWithoutExtension(inputFile) + "_features_data.json";
            }

            DiscreteSignal inputSignal = Tools.ReadSignalFromMp3(inputFile);
            Features f = new Features(inputSignal);
            FeaturesData fdata = f.GetFeaturesData();

            using (StreamWriter file = File.CreateText(outFile))
            {
                JsonSerializer serializer = new JsonSerializer();
                serializer.Serialize(file, fdata);
            }
        }

        /// <summary>
        /// Extracts and saves to JSON features from each mp3-file in folder.
        /// </summary>
        /// <param name="folder">Folder to process.</param>
        public static void ExtractFeaturesFromMP3InFolder(string folder)
        {
            DirectoryInfo d = new DirectoryInfo(folder);
            FileInfo[] Files = d.GetFiles("*.mp3");

            string folderName = d.FullName + @"\JsonData\";
            
            Directory.CreateDirectory(folderName);
            
            foreach (FileInfo f in Files)
            {
                string outFile=folderName+ Path.GetFileNameWithoutExtension(f.Name) + "_features_data.json";
                ExtractFeaturesAndSaveToJSON(f.FullName, outFile);
            }
        }
    }
}
