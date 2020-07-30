using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NWaves.Signals;

namespace FeaturesExtraction
{
    // for key and tonality use KeyTonality class
    // for computation, prepared NWaves.DiscreteSignal is recommended. 

    // static classes for other operations also provided

    /// <summary>
    /// Computes Key and Tonality from DiscreteSignal
    /// </summary>
    public class KeyTonality
    {
        /// <summary>
        /// Key of discrete signal.
        /// </summary>
        public Keys Key { get; private set; }
        /// <summary>
        /// Tonality vector of discrete signal.
        /// </summary>
        public float[] TonalityVector { get; private set; }
        /// <summary>
        /// Indicates if computation is processed.
        /// </summary>
        public bool Processed { get; private set; } = false;

        DiscreteSignal Signal { get; set; }
        
        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="signal"></param>
        public KeyTonality(DiscreteSignal signal)
        {
            Signal = signal;
        }

        /// <summary>
        /// Computes key and tonality from provided signal.
        /// </summary>
        public void ProcessKeyTonality()
        {
            if (Processed)
            {
                return;
            }

            
            Spectrogram1 spectrogramClass = new Spectrogram1(Signal, 2048, 512);
            spectrogramClass.Process();

            var spectrogram = spectrogramClass.Spectrogram;
            var summedChromagram = LogFrequencySpectrogramAndChromagram.SummedChromagramFromSpectrogram(spectrogram,spectrogramClass.SampleRate);

            TonalityVector = Tonality.TonalityFromChromagram(summedChromagram);
            for (int i = 0; i < TonalityVector.Length; i++)
            {
                TonalityVector[i] = TonalityVector[i] / Signal.Samples.Length;
            }
            Key = Tonality.KeyFromTonality(TonalityVector);
            Processed = true;
        }
    }




    //----------------------------------------------------------------------------------//
    // Classes to process computation to determine key and tonality from spectrogram.   //
    //----------------------------------------------------------------------------------//

    #region static computational classes for key and tonality



    /// <summary>
    /// Music keys.
    /// </summary>
    public enum Keys {
#pragma warning disable 1591
        [Description("C Major")] CDur,
        [Description("C# Major")] CisDur,
        [Description("D Major")] DDur,
        [Description("D# Major")] DisDur,
        [Description("E Major")] EDur,
        [Description("F Major")] FDur,
        [Description("F# Major")] FisDur,
        [Description("G Major")] GDur,
        [Description("G# Major")] GisDur,
        [Description("A Major")] ADur,
        [Description("A# Major")] AisDur,
        [Description("B Major")] BDur,

        [Description("C Minor")] CMoll,
        [Description("C# Minor")] CisMoll,
        [Description("D Minor")] DMoll,
        [Description("D# Minor")] DisMoll,
        [Description("E Minor")] EMoll,
        [Description("F Minor")] FMoll,
        [Description("F# Minor")] FisMoll,
        [Description("G Minor")] GMoll,
        [Description("G# Minor")] GisMoll,
        [Description("A Minor")] AMoll,
        [Description("A# Minor")] AisMoll,
        [Description("B Minor")] BMoll,
#pragma warning restore 1591
    };

    /// <summary>
    /// Static class to compute:
    /// >> log-frequency (pitch) spectrogram from spectrogram
    /// >> chromagram
    /// >> summed chromagram
    /// >> pitch frequencies from pitch number
    /// >> note names (pitch names) from pitch numbers
    /// 
    /// based on article https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S1_SpecLogFreq-Chromagram.html
    /// (frequency of pitch and note name only rewritten to C#, pitch spectrogram and chromagram inspired.
    /// </summary>
    public static class LogFrequencySpectrogramAndChromagram
    {
        //based on article https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S1_SpecLogFreq-Chromagram.html

        /// <summary>
        /// Computes center frequency of MIDI pitch.
        /// Reference pitch 69 equals to 440 Hz.
        /// </summary>
        /// <param name="pitch"></param>
        /// <returns></returns>
        public static double FrequencyOfPitch(double pitch)
        {
            double pitch_ref = 69;  //reference pitch
            double freq_ref = 440;  //reference frequency
            return Math.Pow(2, ((pitch - pitch_ref) / 12.0)) * freq_ref;
        }

        /// <summary>
        /// Returns name of the note (eg. A1, C#3).
        /// Reference pitch 69 equals to 440 Hz.
        /// </summary>
        /// <param name="pitch">Pitch value</param>
        /// <returns></returns>
        public static string NoteName(int pitch)
        {
            var chromas = new string[] { "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#" };
            var mod = (((pitch - 69) % 12) + 12) % 12;  //we need modulo in positive integers
            string name = chromas[mod] + (pitch / 12 - 1).ToString(); //using feature that integer division returns floored result
            return name;
        }

        /// <summary>
        /// Computes log-frequency spectrogram from linear-frequency spectrogram.
        /// Log-frequency axis is measured in pitches.
        /// Reference pitch 69 equals to 440 Hz.
        /// </summary>
        /// <param name="spectrogram">linear-frequency spectrogram</param>
        /// <param name="sampleRate">Sample rate of the song.</param>
        /// <returns></returns>
        public static List<float[]> LogFrequencySpectrogramFromSpectrogram(List<float[]> spectrogram, float sampleRate)
        {
            List<float[]> lfSpectrogram = new List<float[]>();

            //for each frame in original spectrogram we compute new frame of log-frequency spectrogram
            foreach (var originalFrame in spectrogram)
            {
                var lfFrame = new float[128];

                //for each pitch in lfFrame we sum up corresponding frequencies in original frame
                for (int pitch = 0; pitch < lfFrame.Length; pitch++)
                {
                    var bottomFrequency = FrequencyOfPitch(pitch - 0.5);
                    var topFrequency = FrequencyOfPitch(pitch + 0.5);

                    // for the bin number 'k', its frequency is k*sampling_frequency/fftSize 
                    int bottomIndex = (int)Math.Floor(bottomFrequency * 2 * originalFrame.Length / sampleRate);
                    int topIndex = (int)Math.Ceiling(topFrequency*2*originalFrame.Length/sampleRate);

                    lfFrame[pitch] = SumArrayIndices(originalFrame,
                        bottomIndex, topIndex);
                }
                lfSpectrogram.Add(lfFrame);
            }

            return lfSpectrogram;
        }

        /// <summary>
        /// Computes sum of array values. For indices in (minIndex,maxIndex).
        /// </summary>
        /// <param name="array"></param>
        /// <param name="minIndex">Indices lower bound.</param>
        /// <param name="maxIndex">Indices upper bound.</param>
        /// <returns></returns>
        static float SumArrayIndices(float[] array, int minIndex, int maxIndex)
        {
            var min = Math.Max(0, minIndex); //Set the first index to be counted.
            var max = Math.Min(array.Length-1, maxIndex);         //Set upper bound.
            float sum = 0;

            for (int i = min; i <= max; i++)
            {
                sum += array[i];
            }

            return sum;
        }


        /// <summary>
        /// Computes chromagram (bins corresponding to 12 semitones of musical octave) for each frame 
        /// in pitch (log-frequency) spectrogram.
        /// </summary>
        /// <param name="lfSpectrogram">pitch (log-frequency) spectrogram</param>
        /// <returns></returns>
        public static List<float[]> ChromagramFromLfSpectrogram(List<float[]> lfSpectrogram)
        {
            List<float[]> chromagram = new List<float[]>();

            //for each frame in pitch spectrogram sum corresponding pitches. (eg. C=...+C-1+C0+C1+C2+...)
            foreach (var lfFrame in lfSpectrogram)
            {
                var sum = new float[12]; //12 basic tones (C, C#,...), C has index 0

                //for each pitch in frame add value to corresponding bracket
                for (int pitch = 0; pitch < lfFrame.Length; pitch++)
                {
                    int note = pitch % 12;
                    sum[note] += lfFrame[pitch];
                }

                chromagram.Add(sum);
            }

            return chromagram;
        }

        /// <summary>
        /// Computes chromagram (bins corresponding to 12 semitones of musical octave) for each frame 
        /// in spectrogram.
        /// 0 corresponds to C, 1 corresponds to C#, etc...
        /// </summary>
        /// <param name="spectrogram">spectrogram</param>
        /// <param name="sampleRate">Sample rate of the song.</param>
        /// <returns></returns>
        public static List<float[]> ChromagramFromSpectrogram(List<float[]> spectrogram, float sampleRate)
        {
            var lfSpectrogram = LogFrequencySpectrogramFromSpectrogram(spectrogram, sampleRate);
            return ChromagramFromLfSpectrogram(lfSpectrogram);
        }

        /// <summary>
        /// Computes summed chromagram from chromagram.
        /// (Summes up all chroma frames to one.)
        /// 0 corresponds to C, 1 corresponds to C#, etc...
        /// </summary>
        /// <param name="chromagram"></param>
        /// <returns></returns>
        public static float[] SummedChromagramFromChromagram(List<float[]> chromagram)
        {
            float[] sum = new float[12];

            //for each frame in chromagram we add it to corresponding notes.
            foreach (var frame in chromagram)
            {
                for (int note = 0; note < frame.Length; note++)
                {
                    sum[note] += frame[note];
                }
            }

            return sum;
        }


        /// <summary>
        /// Computes summed chromagram from spectrogram.
        /// (Summes up all chroma frames to one.)
        /// </summary>
        /// <param name="spectrogram">spectrogram</param>
        /// <param name="sampleRate">Sample rate of the song.</param>
        /// <returns></returns>
        public static float[] SummedChromagramFromSpectrogram(List<float[]> spectrogram, float sampleRate)
        {
            var chromagram = ChromagramFromSpectrogram(spectrogram,sampleRate);
            return SummedChromagramFromChromagram(chromagram);
        }
    }

    /// <summary>
    /// Static class to compute key by tonality score for each key.
    /// </summary>
    public static class Tonality
    {
        /// <summary>
        /// Krumhansl's major key profile vector.
        /// Determines distribution of individual notes in key (starting with tonic on first position).
        /// </summary>
        static float[] majorProfile = { 6.35f, 2.23f, 3.48f, 2.33f, 4.38f, 4.09f, 2.52f, 5.19f, 2.39f, 3.66f, 2.29f, 2.88f };
        /// <summary>
        /// Krumhansl's minor key profile vector
        /// Determines distribution of individual notes in key (starting with tonic on first position).
        /// </summary>
        static float[] minorProfile = { 6.33f, 2.68f, 3.52f, 5.38f, 2.60f, 3.53f, 2.54f, 4.75f, 3.98f, 2.69f, 3.34f, 3.17f };

        static readonly List<float[]> keyProfileMatrix = ComputeKeyProfileMatrix();

        /// <summary>
        /// Shifts array right (eg. [1,2,3,4,5] by 1 -> [5,1,2,3,4]).
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <param name="number">number of positions to shift array</param>
        /// <returns></returns>
        static T[] ShiftArray<T>(T[] array, int number)
        {
            var newArray = new T[array.Length];
            for (int i = 0; i < array.Length; i++)
            {
                newArray[(i + number) % newArray.Length] = array[i];
            }

            return newArray;
        }

        /// <summary>
        /// Computes Krumhansl's key profile matrix using key profile vectors.
        /// 0-11 collumns of matrix corresponds to C Major, C# Major, ... , B Major key,
        /// 12-23 collumns of matrix corresponds to C Minor, C# Minor, ... , B Minor key.
        /// (provided refference pitch 69 is A 440 Hz)
        /// </summary>
        /// <returns></returns>
        static List<float[]>ComputeKeyProfileMatrix()
        {
            //normalize to same scale as major profile
            for (int i = 0; i < minorProfile.Length; i++)
            {
                minorProfile[i] *= 41.79f / 44.51f;
            }

            var matrix = new List<float[]>();

            for (int i = 0; i < 12; i++)
            {
                matrix.Add(ShiftArray(majorProfile, i));
            }
            for (int i = 0; i < 12; i++)
            {
                matrix.Add(ShiftArray(minorProfile, i));
            }

            return matrix;
        }

        /// <summary>
        /// Scalar product of 2 vectors.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        static float VectorDotProduct(float[] a, float[] b)
        {
            if (a.Length != b.Length) throw new ArgumentException("Vectors have not the same length.");

            float result = 0;
            for (int i = 0; i < a.Length; i++)
            {
                result += a[i] * b[i];
            }

            return result;
        }

        /// <summary>
        /// Computes tonality profile from given summed chromagram. (Format float[12])
        /// </summary>
        /// <param name="summedChromagram">Summed chromagram (in form of float[12]).</param>
        /// <returns>Tonality profile (format float[24])</returns>
        public static float[] TonalityFromChromagram (float[] summedChromagram)
        {
            if (summedChromagram.Length != 12) throw new ArgumentException("For computing tonality, summed chromagram" +
                " of 12 elements is needed. Provided array does not have 12 elements.");

            var tonality = new float[24];

            for (int i = 0; i < tonality.Length; i++)
            {
                tonality[i] = VectorDotProduct(summedChromagram, keyProfileMatrix[i]);
            }

            return tonality;
        }

        /// <summary>
        /// Computes key from given tonality.
        /// (selects the biggest value).
        /// </summary>
        /// <param name="tonality">Tonality (in format float[24]).</param>
        /// <returns>Key.</returns>
        public static Keys KeyFromTonality(float[] tonality)
        {
            if (tonality.Length != 24) throw new ArgumentException("Tonality of wrong length given. Tonality must have 24 elements -" +
                  " for each key C Major, C# Major,...,C Minor,...B Minor one element.");

            var index = Array.IndexOf(tonality, tonality.Max());

            return (Keys)index;
        }

    }

    #endregion


    class Spectrogram1
    {
        public int SampleRate { get; private set; }
        public int FFTSize { get; private set; }
        public int Step { get; private set; }
        DiscreteSignal Signal { get; set; }

        public List<float[]> Spectrogram = new List<float[]>();
        
        public Spectrogram1(DiscreteSignal signal, int fftSize, int step)
        {
            Signal = signal;
            SampleRate = Signal.SamplingRate;
            FFTSize = fftSize;
            Step = step;
        }

        

        public void Process()
        {
            int processed = 0;            

            while (Signal.Samples.Length - processed - 1> FFTSize)
            {
                //---------------//
                NAudio.Dsp.Complex[] fft_buffer = new NAudio.Dsp.Complex[FFTSize];                
                
                for (int i = 0; i < FFTSize; i++)
                {
                    fft_buffer[i].X = Signal.Samples[processed +i];
                    fft_buffer[i].Y = 0;
                    fft_buffer[i].X *= (float)NAudio.Dsp.FastFourierTransform.HammingWindow(i, FFTSize);
                }
                processed += Step;
                NAudio.Dsp.FastFourierTransform.FFT(true, (int)Math.Log(FFTSize, 2.0), fft_buffer);

                float[] fftNAudio = new float[FFTSize / 2];
                for (int i = 0; i < fftNAudio.Length; i++)
                {
                    var fftL = fft_buffer[i];
                    var fftR = fft_buffer[fft_buffer.Length - i - 1];

                    // note that this is different than just taking the absolute value
                    float absL = (float)Math.Sqrt(fftL.X * fftL.X + fftL.Y * fftL.Y);
                    float absR = (float)Math.Sqrt(fftR.X * fftR.X + fftR.Y * fftR.Y);
                    fftNAudio[i] = (absL + absR) / 2;
                }
                Spectrogram.Add(fftNAudio);
            }
        }
    }
}
