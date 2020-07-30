using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using NWaves.Signals;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace FeaturesExtraction
{
    /// <summary>
    /// Static class to provide tools that helps with audio processing.
    /// </summary>
    public static class Tools
    {
        /// <summary>
        /// Returns IEEE single floating point signal from input file.
        /// If input is stereo, than optionaly converts to mono
        /// </summary>
        /// <param name="inFileInMp3"></param>
        /// /// <param name="convertToMono"></param>
        /// <returns></returns>
        public static DiscreteSignal ReadSignalFromMp3(string inFileInMp3, bool convertToMono=true)
        {
            var reader = new NAudio.Wave.AudioFileReader(inFileInMp3);
            return ReadSignalFromNAudioReader(reader, convertToMono);
        }

        /// <summary>
        /// Returns IEEE single floating point signal from input file.
        /// If input is stereo, than optionaly converts to mono 
        /// </summary>
        /// <param name="reader"></param>
        /// <param name="convertToMono"></param>
        /// <returns></returns>
        public static DiscreteSignal ReadSignalFromNAudioReader(NAudio.Wave.AudioFileReader reader, bool convertToMono = true)
        {
            ISampleProvider provider;

            // if input is in stereo, then convert to mono
            if (convertToMono && reader.WaveFormat.Channels == 2)
            {
                // convert our stereo ISampleProvider to mono
                provider = new StereoToMonoSampleProvider(reader);

            }
            else
            {
                provider = reader;
            }



            List<float> floatSignal = new List<float>();

            int requested = 1024;
            int check = 0;

            // read from provider until we reach end of signal
            do
            {
                float[] buffer = new float[requested];
                check = provider.Read(buffer, 0, requested);
                if (check < requested)
                {
                    var newBuffer = new float[check];
                    for (int i = 0; i < newBuffer.Length; i++)
                    {
                        newBuffer[i] = buffer[i];
                    }
                    buffer = newBuffer;
                }
                floatSignal.AddRange(buffer);
            } while (requested == check);

            DiscreteSignal discreteSignal = new DiscreteSignal(provider.WaveFormat.SampleRate, floatSignal);
            return discreteSignal;
        }

        /// <summary>
        /// Creates signal with clap sound on specified time stamps.
        /// </summary>
        /// <param name="timeBeats">Time stamps.</param>
        /// <param name="clapFile">mp3 file with clap sound</param>
        /// <returns></returns>
        public static ISampleProvider ClappingToBeats(float[] timeBeats, string clapFile)
        {
            DiscreteSignal d = Tools.ReadSignalFromMp3(clapFile);

            int length = (int)(timeBeats[timeBeats.Length - 1] * d.SamplingRate + 1) + d.Samples.Length;

            float[] sound = new float[length];

            foreach(var time in timeBeats)
            {
                //var t = (int)(time * d.SamplingRate);
                Array.Copy(d.Samples, 0, sound, (int)(time * d.SamplingRate), d.Samples.Length);
            }

            ISampleProvider provider = new SampleProvider(sound, d.SamplingRate);

            return provider;
        }
        
    }

    /// <summary>
    /// Auxiliary class to create ISampleProvider from samples and sample rate.
    /// </summary>
    class SampleProvider : ISampleProvider
    {
        float[] Samples { get; }
        int Pointer { get; set; }
        public WaveFormat WaveFormat { get; }


        public SampleProvider(float[] samples, int sampleRate)
        {
            WaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
            Samples = samples;
            Pointer = 0;
        }

        /// <summary>
        /// Fill the specified buffer with 32 bit floating point samples
        /// </summary>
        /// <param name="buffer">The buffer to fill with samples.</param>
        /// <param name="offset">Offset into buffer</param>
        /// <param name="count">The number of samples to read</param>
        /// <returns>the number of samples written to the buffer.</returns>
        public int Read(float[] buffer, int offset, int count)
        {
            int processedCount = count;
            if (Pointer + count > Samples.Length)
            {
                processedCount = Samples.Length - Pointer;
            }

            for (int i = 0; i < processedCount; i++)
            {
                buffer[offset + i] = Samples[Pointer];
                Pointer++;
            }
            return processedCount;
        }
    }

    /// <summary>
    /// Tools that helps with debugging and verifying result.
    /// </summary>
    public static class RandomTools
    {
        /// <summary>
        /// Plays audio from SampleProvider. 
        /// </summary>
        /// <param name="signal"></param>
        public static void PlaySampleProvider(ISampleProvider signal)
        {
            using (var wo = new WaveOutEvent())
            {
                wo.Init(signal);
                wo.Play();
                while (wo.PlaybackState == PlaybackState.Playing)
                {
                    Thread.Sleep(500);
                }
            }
        }
        /// <summary>
        /// Plays audio from MP3 file.
        /// </summary>
        /// <param name="inputFile">MP3 file.</param>
        public static void PlayMP3File(string inputFile)
        {
            WaveStream mainOutputStream = new Mp3FileReader(inputFile);
            WaveChannel32 volumeStream = new WaveChannel32(mainOutputStream);

            WaveOutEvent player = new WaveOutEvent();

            player.Init(volumeStream);

            player.Play();
        }

    }
}
