using CommandLine;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using NWaves.Signals;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading;

namespace FeaturesExtraction
{
    class Program
    {


        static void Main(string[] args)
        {
            // add claps on computed beat times, extract features or extract features and saves them to JSON
            //string inputFile = "data_files/test.mp3";
            //ProgramOutput.AddClapsOnRhytm(inputFile);
            //ProgramOutput.ExtractFeatures(inputFile, true);
            //ProgramOutput.ExtractFeaturesAndSaveToJSON(inputFile);


            // create DiscreteSignal
            //var inputFile = "data_files/test.mp3";
            //DiscreteSignal inputSignal = Tools.ReadSignalFromMp3(inputFile);

            // create spectrogram from input signal
            //Spectrogram1 spec = new Spectrogram1(inputSignal, inputSignal.SamplingRate, 512);
            //spec.Process();
            //var spectrogram2 = spec.Spectrogram;

            // Process whole folder to extract features
            //string inFolder = "mp3_input_full";
            //ProgramOutput.ExtractFeaturesFromMP3InFolder(inFolder);
            
        }
    }
}

