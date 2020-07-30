using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace FeaturesExtraction
{
    static class Examples
    {
        /// <summary>
        /// computes envelope and tempo estimation from file
        /// </summary>
        static void ComputeGlobalTempoEstimate()
        {
            string inputFile = "mozart.mp3";            

            NWaves.Signals.DiscreteSignal inputSignal = Tools.ReadSignalFromMp3(inputFile);

            var envelope = new OnsetStrengthEnvelope(inputSignal);
            envelope.ProcessEnvelope();

            /*
            // write envelope
            foreach(var x in envelope.Envelope)
            {
                Console.WriteLine(x);
            }
            */


            var tempoEstimate = new GlobalTempoEstimate(envelope.Envelope, envelope.EnvelopeSampleRate);
            tempoEstimate.ProcessTempoEstimation();
            
            // write autocorrelation
            foreach (var x in tempoEstimate.AutoCorrelation)
            {
                Console.WriteLine(x);
            }
        }

        /// <summary>
        /// Creates sine wave and plays it.
        /// </summary>
        static void SineWave()
        {
            var sine20Seconds = new SignalGenerator()
            {
                Gain = 0.8,
                Frequency = 440,
                Type = SignalGeneratorType.Sin
            }.Take(TimeSpan.FromSeconds(20));

            using (var wo = new WaveOutEvent())
            {
                wo.Init(sine20Seconds);
                wo.Play();
                while (wo.PlaybackState == PlaybackState.Playing)
                {
                    Thread.Sleep(500);
                }
            }
        }

        /// <summary>
        /// Writes values from MEL mapping matrix to console.
        /// Used to plot mapping triangles.
        /// </summary>
        static void MelMappingTriangles()
        {
            var f = MelFilters.MelMappingMatrix(40, 2048, 8000, 20);

            for (int j = 0; j < f[0].Length; j++)
            {
                Console.Write(j + 20 + " ");
                for (int i = 0; i < 40; i++)
                {
                    Console.Write(f[i][j] + " ");
                }
                Console.WriteLine();
            }
        }

        
    }
}
