using NAudio.Wave;
using NWaves;
using NWaves.Signals;
using NWaves.Windows;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FeaturesExtraction
{
    // based on https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf
    // for beat times use DynamicBeatTracking class

    // for computation, prepared NWaves.DiscreteSignal is recommended. 
    


    #region Onset strength envelope
    /// <summary>
    /// Class that computes Onset strength envelope according to https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf
    /// </summary>
    public class OnsetStrengthEnvelope
    {   
        /// <summary>
        /// Signal to process.
        /// </summary>
        DiscreteSignal Signal { get; set; }
        /// <summary>
        /// Computed envelope. Computed via ProcessEnvelope().
        /// </summary>
        public float[] Envelope { get; private set; }

        /// <summary>
        /// How many values in envelope is in 1 second of input signal
        /// </summary>
        public int EnvelopeSampleRate { get; } = 250;

        /// <summary>
        /// Indicates if envelope is processed.
        /// </summary>
        public bool Processed { get; private set; } = false;
        

        /// <summary>
        /// Constructor of the class.
        /// </summary>
        /// <param name="signal">Signal to process.</param>
        public OnsetStrengthEnvelope(DiscreteSignal signal)
        {
            Signal = signal;
        }

        /// <summary>
        /// Void to process envelope from input file. 
        /// Follows https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf in section 3.1
        /// </summary>
        public void ProcessEnvelope()
        {
            if (Processed)
            {
                return;
            }

            // resample input to 8 kHz            
            var resampledSignal = ResampleTo8kHzNWaves();
            // computes STFT spectrogram
            var spectrogram = STFTspectrogram(resampledSignal);
            // converts to auditory representation by mapping it to 40 MEL bands
            var melBands = MapSpectrogramTo40MELBands(spectrogram);
            // converts values to dB
            var dBSpectrogram = DBFilters.ConvertMelSpectrogramToDB(melBands);
            // computes first order difference allong time in each band, negative values set to 0
            var dBSpectrogramFirstOrderDifference = FirstOrderDifference(dBSpectrogram, true);
            // differences are summed across all frequency bands
            var summedSignalAcrossAllFrequencies = SumRowsInMatrix(dBSpectrogramFirstOrderDifference);
            var signal = summedSignalAcrossAllFrequencies;
            // this signal is passed through high pass filter with a cutoff at 0.4 Hz
            var filteredSignal = FrequencyFilters.HighPassFilter(signal, 8000, 0.4f, 1); // #DANGER - bandwidth on last position - not sure what pass there
            // signal is smoothed by convolving with Gaussian envelope 20 ms wide
            var smoothedSignal = SmoothingFilters.GaussianConvolution(filteredSignal, 160); // 20ms window in 8000 samples per 
                                                                                            // second means 160 samples in one window
            // smoothed signal is normalized by dividing it by its standard deviation
            NormalizeEnvelope(smoothedSignal);

            Envelope = smoothedSignal;
            Processed = true;
        }

        /// <summary>
        /// Reads input signal and return resampled IEEE single floating point signal.
        /// </summary>
        /// <returns></returns>
        DiscreteSignal ResampleTo8kHzNWaves()
        {   
            var resampler = new NWaves.Operations.Resampler();
            var resampledSignal = resampler.Resample(Signal, 8000);

            return resampledSignal;            
        }

        

        /// <summary>
        /// Computes spectrogram from resampled 8kHz IEEE single floating point signal.
        /// </summary>
        /// <param name="resampled8kHzIEEEfloatSignal">signal with 8 kHz samplerate</param>
        /// <returns></returns>
        static List<float[]> STFTspectrogram(DiscreteSignal resampled8kHzIEEEfloatSignal)
        {
            //fourier transfomr should be on 32ms windows and 4ms steps between frames according to paper
            //with 8000 samples/s it is 0.032*8000=256 samples windows and 0.004*8000=32 samples step.            
            var spectrogramClass = new Spectrogram1(resampled8kHzIEEEfloatSignal, 256, 32);
            spectrogramClass.Process();
            
            return spectrogramClass.Spectrogram;
        }

        /// <summary>
        /// Computes 40 Mel bands spectrogram from provided frequency spectrogram.
        /// </summary>
        /// <param name="spectrogram">spectrogram to map to mel bands</param>
        /// <returns></returns>
        static List<float[]> MapSpectrogramTo40MELBands(List<float[]> spectrogram)
        {
            // filter matrix to apply to spectrogram
            var filterMatrix = MelFilters.MelMappingMatrix(40, spectrogram[0].Length, 8000, 0);

            List<float[]> melSpectrogram = new List<float[]>();

            // for each frame (frequency distribution at time interval) computes MEL frame using filter matrix
            foreach(var frame in spectrogram)
            {
                var mels = new float[40];
                // each MEL band is computed by convolution of filter and frequency frame
                for (int i = 0; i < mels.Length; i++)
                {
                    mels[i] = ScalarProduct(frame, filterMatrix[i]);
                }
                melSpectrogram.Add(mels);
            }

            return melSpectrogram;
        }

        /// <summary>
        /// Returns scalar product of 2 vectors of the same length.
        /// </summary>
        /// <param name="vector1"></param>
        /// <param name="vector2"></param>
        /// <returns></returns>
        static float ScalarProduct (float[] vector1, float[] vector2)
        {
            if (vector1.Length!=vector2.Length)
            {
                throw new ArgumentException("Format of vectors do not match.");
            }
            float sum = 0;
            for (int i = 0; i < vector1.Length; i++)
            {
                sum += vector1[i] * vector2[i];
            }
            return sum;
        }

        /// <summary>
        /// Computes first order difference in matrix collumns.
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="negativeToZero">If true, then negative values are set to 0</param>
        /// <returns></returns>
        static List<float[]> FirstOrderDifference (List<float[]> matrix, bool negativeToZero = false)
        {
            if (matrix.Count==0)
            {
                return new List<float[]>();
            }

            List<float[]> differentiateMatrix = new List<float[]>();

            //we add first column of matrix
            float[] firstVector;
            firstVector = matrix[0].Select(x =>
            { if (negativeToZero) { return Math.Max(0, x); } else { return x; } }).ToArray();
            differentiateMatrix.Add(firstVector);
            
            // we subtract collumn vector in matrix from the next collumn vector
            for (int vector = 1; vector < matrix.Count; vector++)
            {
                float[] differentialVector;

                // subtracting vectors
                differentialVector = matrix[vector].Zip(matrix[vector - 1], (x, y) => 
                    { if (negativeToZero) {return Math.Max(0, x - y);}else { return x - y; } }).ToArray();

                differentiateMatrix.Add(differentialVector);
            }

            return differentiateMatrix;
        }

        /// <summary>
        /// Returns sum of all rows in matrix.
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        static float[] SumRowsInMatrix (List<float[]> matrix)
        {
            float[] sumRow = new float[matrix.Count];
            for (int i = 0; i < matrix.Count; i++)
            {
                sumRow[i] = matrix[i].Sum();
            }
            return sumRow;
        }

        /// <summary>
        /// Normalizes vector by dividing each value by standard deviation.
        /// </summary>
        /// <param name="envelope"></param>
        void NormalizeEnvelope(float[] envelope)
        {
            float standardDeviation = StandardDeviation(envelope);
            for (int i = 0; i < envelope.Length; i++)
            {
                //divide with its standard deviation
                envelope[i] /= standardDeviation;
            }
        }

        /// <summary>
        /// Computes standard deviation from entry.
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        float StandardDeviation(float[] values)
        {
            // mean value
            var mean = values.Average();
            // sum of squared differences (x-E(x))^2
            var sumOfSquaredDifference = values.Sum(x => (x - mean) * (x - mean));

            var standardDeviation =(float) Math.Sqrt(sumOfSquaredDifference / values.Length);
            return standardDeviation;
        }

    }

    /// <summary>
    /// Class to provide smoothing filters.
    /// </summary>
    public static class SmoothingFilters
    {
        /// <summary>
        /// Convolutes signal with gaussian filter of requested window length.
        /// Applies windows not-overlapping.
        /// </summary>
        /// <param name="signal">signal to convolute</param>
        /// <param name="windowLength">length of filter window</param>
        /// <returns></returns>
        public static float[] GaussianConvolution(float[] signal, int windowLength)
        {
            // https://github.com/ar1st0crat/NWaves
            var gaussFilter = NWaves.Windows.Window.Gaussian(windowLength);

            float[] convolutedSignal = new float[signal.Length];
            for (int i = 0; i < signal.Length; i++)
            {
                convolutedSignal[i] = signal[i] * gaussFilter[i % gaussFilter.Length];
            }

            return convolutedSignal;
        }
    }

    /// <summary>
    /// Class to provide frequency filters
    /// </summary>
    public static class FrequencyFilters
    {
        /// <summary>
        /// Applies high-pass filter on signal.
        /// </summary>
        /// <param name="signal">Signal to process</param>
        /// <param name="sampleRate">Sample rate of the signal</param>
        /// <param name="cutOffFrequency">cut-off frequency of the filter</param>
        /// <param name="bandwidth">bandwidth of signal in octaves</param>          //extracted from source code of naudio, not sure
        /// <returns></returns>
        public static float[] HighPassFilter(float[] signal, float sampleRate, float cutOffFrequency,float bandwidth)
        {
            // https://github.com/naudio/NAudio/blob/master/NAudio/Dsp/BiQuadFilter.cs
            var filter = NAudio.Dsp.BiQuadFilter.HighPassFilter(sampleRate, cutOffFrequency, bandwidth);

            float[] filteredSignal = new float[signal.Length];
            for (int i = 0; i < signal.Length; i++)
            {
                filteredSignal[i] = filter.Transform(signal[i]);
            }

            return filteredSignal;
        }        
    }

    /// <summary>
    /// Class to provides operations with dB, amplitude and power.
    /// </summary>
    public static class DBFilters
    {
        /// <summary>
        /// Transforms amplitude spectrogram to power spectrogram (each entry value is squared).
        /// </summary>
        /// <param name="amplitudeSpectrogram"></param>
        /// <returns></returns>
        public static List<float[]> AmplitudeSpectrogramToPowerSpectrogram(List<float[]> amplitudeSpectrogram)
        {
            List<float[]> powerSpectrogram = new List<float[]>();
            foreach (var vector in amplitudeSpectrogram)
            {
                float[] powerVector = new float[vector.Length];
                for (int i = 0; i < vector.Length; i++)
                {
                    powerVector[i] = vector[i] * vector[i];                    
                }
                powerSpectrogram.Add(powerVector);
            }
            return powerSpectrogram;
        }

        /// <summary>
        /// Converts amplitude spectrogram to dB spectrogram.
        /// </summary>
        /// <param name="spectrogram"></param>
        /// <param name="referenceEqualsMaximumValue">If true, then referential value (to computes dB) is maximum squared value of entry</param>
        /// <param name="amin">minimal value</param>
        /// <param name="top_db">if greater than 0, then minimal value of dB is set to maximal value minus this</param>
        /// <returns></returns>
        public static List<float[]> ConvertMelSpectrogramToDB(List<float[]> spectrogram, bool referenceEqualsMaximumValue=false, 
            float amin = 1e-10f, float top_db = 80.0f)
        {
            // https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#power_to_db

            List<float[]> dBSpectrogram = new List<float[]>();

            spectrogram = AmplitudeSpectrogramToPowerSpectrogram(spectrogram);

            //minimal value must be positive
            if (amin <= 0) { throw new ArgumentException("minimal value must be positive"); }

            

            float ref_value;
            if (referenceEqualsMaximumValue)
            {
                var spectrogramMaximumValue = (from float[] array in spectrogram select array.Max()).Max();
                ref_value = spectrogramMaximumValue;
            }
            else
            {
                ref_value = 1f;
            }

            foreach (float[] vector in spectrogram)
            {
                var dBVector = new float[vector.Length];
                for (int i = 0; i < vector.Length; i++)
                {
                    dBVector[i] = (float)(10.0 * Math.Log10(Math.Max(amin, vector[i])));
                    dBVector[i] -= (float)(10.0 * Math.Log10(Math.Max(amin, ref_value)));
                }
                dBSpectrogram.Add(dBVector);
            }


            if (top_db > 0)
            {
                var dbMaximum = (from float[] vector in dBSpectrogram select vector.Max()).Max();
                foreach (var dBVector in dBSpectrogram)
                {
                    for (int i = 0; i < dBVector.Length; i++)
                    {
                        dBVector[i] = Math.Max(dBVector[i], dbMaximum - top_db);
                    }
                }
            }
            return dBSpectrogram;
        }
    }

    /// <summary>
    /// Class to provides filters for MEL conversion
    /// </summary>
    public static class MelFilters
    {
        // good article to understand Mel filtering - http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

        /// <summary>
        /// Computes MEL mapping matrix.
        /// </summary>
        /// <param name="numberOfMelBins"></param>
        /// <param name="lengthOfVectors">number of samples in each frame of spectrogram.</param>
        /// <param name="maxFrequency">maximum frequency of spectrogram to be applied to</param>
        /// <param name="minFrequency">minimum frequency of spectrogram to be applied to</param>
        /// <param name="normalizedBandwidth">If true, than triangles' area equals 1.</param>
        /// <param name="formulaToUse"></param>
        /// <returns></returns>
        static public List<float[]> MelMappingMatrix(int numberOfMelBins, int lengthOfVectors ,float maxFrequency, float minFrequency=0,
            bool normalizedBandwidth = true, MelFormulas formulaToUse = MelFormulas.SlaneyFormula)
        {
            List<float[]> matrix = new List<float[]>();
            var coeff = CepstralCoefficientsInMels(numberOfMelBins, HzToMel(minFrequency), HzToMel(maxFrequency));
            var hzCoeffs = MelToHz(coeff,formulaToUse);

            //use numerical errors
            minFrequency = hzCoeffs[0];
            maxFrequency = hzCoeffs[hzCoeffs.Length - 1];

            var step = (maxFrequency - minFrequency) / lengthOfVectors;
            
            //for each bin add triangle filter to matrix
            for (int binNumber = 1; binNumber <= numberOfMelBins; binNumber++)
            {
                //vertices positions in Hz 
                var leftTriangleVertex = hzCoeffs[binNumber - 1];
                var rightTriangleVertex = hzCoeffs[binNumber + 1];
                
                //vertices positions in array bin number
                int leftTriangleArrayBin = (int)Math.Floor((leftTriangleVertex - minFrequency) / step);                
                int rightTriangleArrayBin = (int)Math.Floor((rightTriangleVertex - minFrequency) / step);

                if (rightTriangleArrayBin>=lengthOfVectors)
                {
                    rightTriangleArrayBin = lengthOfVectors - 1;
                }

                matrix.Add(TriangleMapping(lengthOfVectors, leftTriangleArrayBin, rightTriangleArrayBin,
                    normalizedBandwidth));
            }

            return matrix;
        }

        /// <summary>
        /// Returns tringle-shape filter.
        /// </summary>
        /// <param name="arrayLength"></param>
        /// <param name="startingPosition">left triangle vertex</param>
        /// <param name="endingPosition">right triangle vertex</param>
        /// <param name="normalized">if true, then height of the triangle is set such that area of triangle equals 1.
        /// Otherwise peak value is 1.</param>
        /// <returns></returns>
        public static float[] TriangleMapping(int arrayLength, int startingPosition, int endingPosition, bool normalized)
        {
            var triangle = new float[arrayLength];

            if (endingPosition==startingPosition)
            {
                triangle[startingPosition] = 1;
                return triangle;
            }
            if (endingPosition-startingPosition==1)
            {
                triangle[startingPosition] = 0.5f;
                triangle[endingPosition] = 0.5f;
                return triangle;
            }

            for (int i = 0; i < (endingPosition-startingPosition)/2 + 1; i++)
            {
                triangle[startingPosition + i] = i / ((endingPosition - startingPosition) / 2f);
                triangle[endingPosition-i] = i / ((endingPosition - startingPosition) / 2f);
            }

            if (normalized)
            {
                int lengthToNormalize = endingPosition - startingPosition;
                
                for (int i = startingPosition; i <= endingPosition; i++)
                {
                    triangle[i] = 2*triangle[i] / lengthToNormalize;
                }
            }

            return triangle;
        }

        /// <summary>
        /// Formulas to use in computation of Mels.
        /// 
        /// The mel scale is a quasi-logarithmic function of acoustic frequency designed such that perceptually
        /// similar pitch intervals (e.g. octaves) appear equal in width over the full hearing range.
        /// 
        /// Because the definition of the mel scale is conditioned by a finite number of subjective psychoaoustical
        /// experiments, several implementations coexist in the audio signal processing literature.
        /// 
        /// In Slaney implementation, the conversion from Hertz to mel is linear below 1 kHz and logarithmic above 1 kHz.
        /// Hidden Markov Toolkit (HTK) implements mel = 2595.0 * log_10(1.0 + frequency / 700.0).
        /// </summary>
        public enum MelFormulas
        {
            /// <summary>
            /// In Slaney implementation, the conversion from Hertz to mel is linear below 1 kHz and logarithmic above 1 kHz.
            /// </summary>
            SlaneyFormula,
            /// <summary>
            /// Hidden Markov Toolkit (HTK) implements mel = 2595.0 * log_10(1.0 + frequency / 700.0).
            /// </summary>
            HTKFormula
        };

        /// <summary>
        /// Computes Mels from Hertz frequency.
        /// MATLAB function: https://labrosa.ee.columbia.edu/matlab/rastamat/hz2mel.m
        /// Python implementation of MATLAB function: https://librosa.github.io/librosa/_modules/librosa/core/time_frequency.html#hz_to_mel
        /// </summary>
        /// <param name="frequencyInHertz">Frequency in Hertz.</param>
        /// <param name="formulaToUse">[optional argumet] Formula to provide calculation.</param>
        /// <returns></returns>
        static public float HzToMel(float frequencyInHertz, MelFormulas formulaToUse = MelFormulas.SlaneyFormula)
        {
            switch (formulaToUse)
            {
                case MelFormulas.HTKFormula:
                    {
                        double mel = 2595 * Math.Log10(1 + (frequencyInHertz / 700));
                        return (float)mel;
                    }
                case MelFormulas.SlaneyFormula:
                    {
                        // linear part
                        float f_min = 0; // 133.33333;
                        float f_sp = 200 / 3; // 66.66667;

                        // log-scale part
                        float min_log_hz = 1000;    // beginning of log region in Hertz
                        float min_log_mel = (min_log_hz - f_min) / f_sp;  // same in mels - starting mel value for log region

                        float logstep = (float) (Math.Log(6.4) / 27); // step size for log region
 
                        /* the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz 
                         * in 27 steps, and is *almost * the ratio between 1000 Hz and the preceding linear 
                         * filter center at 933.33333 Hz(actually 1000 / 933.33333 = 1.07142857142857 and  
                         * exp(log(6.4) / 27) = 1.07117028749447)
                        */
                        
                        // if frequency is in linear part, use formula for linear part
                        if (frequencyInHertz < min_log_hz)
                        {
                            return (frequencyInHertz - f_min) / f_sp;
                        }
                        // if frequency is in log part, use formula for log part
                        else
                        {
                            // beginning of log region plus value of log region
                            return (float) (min_log_mel + Math.Log(frequencyInHertz / min_log_hz)/logstep);
                        }
                    }
                default:
                    throw new ArgumentException("This type of mel computing formula is not implemented.");
            }
            
        }

        /// <summary>
        /// Computes Mels from Hertz frequency.
        /// MATLAB function: https://labrosa.ee.columbia.edu/matlab/rastamat/hz2mel.m
        /// Python implementation of MATLAB function: https://librosa.github.io/librosa/_modules/librosa/core/time_frequency.html#hz_to_mel
        /// </summary>
        /// <param name="frequenciesInHertz">Frequencies in Hertz to convert.</param>
        /// <param name="formulaToUse">[optional argumet] Formula to provide calculation.</param>
        /// <returns></returns>
        static public float[] HzToMel(float[] frequenciesInHertz, MelFormulas formulaToUse = MelFormulas.SlaneyFormula)
        {
            float[] mels = new float[frequenciesInHertz.Length];
            for (int i = 0; i < frequenciesInHertz.Length; i++)
            {
                mels[i] = HzToMel(frequenciesInHertz[i],formulaToUse);
            }
            return mels;
        }

        /// <summary>
        /// Computes Hertz frequency from mel bins.
        /// MATLAB function: https://labrosa.ee.columbia.edu/matlab/rastamat/mel2hz.m
        /// Python implementation of MATLAB function: https://librosa.github.io/librosa/_modules/librosa/core/time_frequency.html#mel_to_hz
        /// </summary>
        /// <param name="mels">Mel bin.</param>
        /// <param name="formulaToUse">[optional argument] Formula to provida calculation.</param>
        /// <returns></returns>
        static public float MelToHz(float mels, MelFormulas formulaToUse = MelFormulas.SlaneyFormula)
        {            
            switch(formulaToUse)
            {
                case (MelFormulas.HTKFormula):
                    {
                        return (float) (700.0 * (Math.Pow(10,mels / 2595.0) - 1.0));
                    }
                case (MelFormulas.SlaneyFormula):
                    {
                        float f_min = 0.0f;
                        float f_sp = 200.0f / 3f;
                        float freqs = f_min + f_sp * mels;


                        float min_log_hz = 1000.0f;                        // # beginning of log region (Hz)
                        float min_log_mel = (min_log_hz - f_min) / f_sp;   //# same (Mels)
                        float logstep =(float) (Math.Log(6.4) / 27.0);                //# step size for log region


                        if (mels<min_log_mel)
                        {
                            return f_min + f_sp * mels;
                        }
                        else
                        {
                            return (float) (min_log_hz * Math.Exp(logstep * (mels - min_log_mel)));
                        }
                    }
                default:
                    throw new ArgumentException("This type of mel computing formula is not implemented.");
            }
    
    
        }

        /// <summary>
        /// Computes Hertz frequency from mel bins.
        /// MATLAB function: https://labrosa.ee.columbia.edu/matlab/rastamat/mel2hz.m
        /// Python implementation of MATLAB function: https://librosa.github.io/librosa/_modules/librosa/core/time_frequency.html#mel_to_hz
        /// </summary>
        /// <param name="mels">Mel bins.</param>
        /// <param name="formulaToUse">[optional argument] Formula to provida calculation.</param>
        /// <returns></returns>
        static public float[] MelToHz(float[] mels, MelFormulas formulaToUse = MelFormulas.SlaneyFormula)
        {
            float[] hertzs = new float[mels.Length];
            for (int i = 0; i < mels.Length; i++)
            {
                hertzs[i] = MelToHz(mels[i],formulaToUse);
            }
            return hertzs;
        }

        /// <summary>
        /// Computes mel values to apply triangle filter to transform Hz spectrogram to Mel spectrogram.
        /// First bin starts from result[0], ends at result[2] and has peak at result[1].
        /// Second bin starst from result[1], ends at result[3] and has peak at result[2].
        /// And so on..
        /// </summary>
        /// <param name="numberOFBins">Number of requested mel bins.</param>
        /// <param name="minimumMel">Minimum mel frequency.</param>
        /// <param name="maximumMel">Maximum mel frequency.</param>
        /// <returns></returns>
        static float[] CepstralCoefficientsInMels (int numberOFBins, float minimumMel, float maximumMel)
        {
            // we linearly divides scale from minimumMel to maximumMel
            // because first bin needs to start and last bin needs to end, we need to get numberOfBins+2 mel coefficients.
            
            var coeff = new float[numberOFBins + 2];
            coeff[0] = minimumMel;
            coeff[coeff.Length - 1] = maximumMel;
            var step = (maximumMel - minimumMel) / (coeff.Length - 1);
            
            for (int i = 0; i < coeff.Length-1; i++)
            {
                coeff[i] = minimumMel + i * step;
            }

            return coeff;
        }
    }
    #endregion

    #region Global tempo estimate

    /// <summary>
    /// Class that computes estimation of global tempo from onset strength envelope
    /// According to this paper - https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf
    /// </summary>
    public class GlobalTempoEstimate
    {
        /// <summary>
        /// Onset strength envelope provided.
        /// </summary>
        float[] OnsetStrengthEnvelope { get; }
        /// <summary>
        /// How many samples in onset strength envelope per 1 second.
        /// </summary>
        float SampleRate { get; }
        /// <summary>
        /// For how big lag we want to compute autocorrelation.
        /// </summary>
        int MaxLag { get; } = 4;

        /// <summary>
        /// Indicates if Global tempo estimation is processed. 
        /// </summary>
        public bool Processed { get; private set; } = false;

        /// <summary>
        /// Indicates whether AutoCorrelation is computed with weighting function or not.
        /// </summary>
        public bool WeightFunctionUsed { get; private set; } = true;

        /// <summary>
        /// After processing contains autocorrelation data.
        /// </summary>
        public float[] AutoCorrelation { get; private set; } = null;
        /// <summary>
        /// After processing contains position of highest peak in autocorrelation data.
        /// Which represents lag between two envelopes.
        /// </summary>
        public int? IdealInterBeatIntervalInSamples { get; private set; } = null;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="onsetStrengthEnvelope">Onset strength envelope to process.</param>
        /// <param name="sampleRate">How many samples in onset strength envelope per 1 second.</param>
        /// <param name="weightFunctionUse">If weighting function should be used during computation.</param>
        public GlobalTempoEstimate(float[] onsetStrengthEnvelope,float sampleRate, bool weightFunctionUse = true)
        {
            OnsetStrengthEnvelope = onsetStrengthEnvelope;
            SampleRate = sampleRate;
            WeightFunctionUsed = weightFunctionUse;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="envelope">Onset strength envelope class. If not processed then will be processed in constructor.</param>
        /// <param name="weightFunctionUse">If weighting function should be used during computation.</param>
        public GlobalTempoEstimate(OnsetStrengthEnvelope envelope, bool weightFunctionUse = true):
            this(envelope.Envelope,envelope.EnvelopeSampleRate,weightFunctionUse)
        {
            if (!envelope.Processed)
            {
                envelope.ProcessEnvelope();                
            }
            OnsetStrengthEnvelope = envelope.Envelope;
            SampleRate = envelope.EnvelopeSampleRate;
            WeightFunctionUsed = weightFunctionUse;
        }

        /// <summary>
        /// Computes auto correlation and ideal interbeat interval.
        /// </summary>
        public void ProcessTempoEstimation()
        {
            if (Processed)
            {
                return;
            }

            bool useWeightingFunction = WeightFunctionUsed;
            ComputeAutoCorrelation(useWeightingFunction);
            ComputeIdealInterBeatInterval();
            Processed = true;
        }

        /// <summary>
        /// Finds index corresponding to highest peak.
        /// </summary>
        void ComputeIdealInterBeatInterval()
        {
            var index = Array.IndexOf(AutoCorrelation, AutoCorrelation.Max());
            IdealInterBeatIntervalInSamples = index;
        }

        /// <summary>
        /// Computes autocorrelation and assigns it to public property.
        /// </summary>
        /// <param name="useWeightingFunction"></param>
        void ComputeAutoCorrelation(bool useWeightingFunction)
        {
            //correlation is computed for lag up to MaxLag seconds -> sampleRate*MaxLag
            float[] autoCorrelation = new float[(int)SampleRate * MaxLag];

            for (int i = 0; i < autoCorrelation.Length; i++)
            {
                float weight = 1;
                if (useWeightingFunction)
                {
                    weight = WeightingFunction(i);
                }
                autoCorrelation[i] = weight * AutoCorrelationWithInterval(i);
            }

            AutoCorrelation = autoCorrelation;
        }

        /// <summary>
        /// Computes value of convolution of onset strength envelope with itself delayed by interval.
        /// </summary>
        /// <param name="numberOfArrayStepsInInterval">How many samples to delay envelope.</param>
        /// <returns></returns>
        float AutoCorrelationWithInterval(int numberOfArrayStepsInInterval)
        {
            double sum=0;
            
            // Assumption that length of envelope is at least 3 times bigger than maximum lag. 
            for (int i = numberOfArrayStepsInInterval; i < Math.Min(OnsetStrengthEnvelope.Length,3*SampleRate*MaxLag); i++)
            {
                sum += OnsetStrengthEnvelope[i] * OnsetStrengthEnvelope[i - numberOfArrayStepsInInterval];
            }

            return (float)sum;
        }

        /// <summary>
        /// Computes perceptual weight of tempo interval
        /// According to human bias to 120 bpm.
        /// </summary>
        /// <param name="numberOfArrayStepsInInterval"></param>
        /// <returns></returns>
        float WeightingFunction(int numberOfArrayStepsInInterval)
        {
            // constants from paper experimental results
            float tau_0 = 0.5f * SampleRate;
            float delta_tau = 1.4f;
            // page 8, equation 6 of https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf
            return (float) Math.Exp(-0.5 * Math.Pow(Math.Log(numberOfArrayStepsInInterval/tau_0, 2)/delta_tau,2));
        }
    }


    #endregion

    #region Dynamic programming of beat tracking

    /// <summary>
    /// Class that computes actual beats. From discrete signal or from provided OnsetStrengthEnvelope and TempoEstimate. 
    /// </summary>
    public class DynamicBeatTracking
    {
        OnsetStrengthEnvelope Envelope { get; }
        GlobalTempoEstimate TempoEstimate { get; }

        /// <summary>
        /// Ideal interbeat interval from GlobalTempoEstimate
        /// </summary>
        int IdealTempo { get; set; }
        /// <summary>
        /// Coefficient to balance terms in computation.
        /// </summary>
        float Alpha { get; set; } = 680;  // optimum according to paper is 680

        /// <summary>
        /// Dynamic table to be filled
        /// </summary>
        float[] Scoring { get; set; }
        /// <summary>
        /// Table of actual preceding beats that gives the best score.
        /// </summary>
        int[] Predestors { get; set; }

        /// <summary>
        /// Time (in samples) where beats are.
        /// </summary>
        public int[] Beats { get; private set; }

        /// <summary>
        /// In samples.
        /// </summary>
        public float AverageBeatInterval { get; private set; }
        /// <summary>
        /// In samples.
        /// </summary>
        public float StandardInterBeatDeviation { get; private set; }
        /// <summary>
        /// In beats per minute.
        /// </summary>
        public float OverallTempoBPM { get; private set; }

        /// <summary>
        /// Sample rate of beats.
        /// </summary>
        public int SampleRate => Envelope.EnvelopeSampleRate;

        /// <summary>
        /// If Dynamic beat tracking is processed.
        /// </summary>
        public bool Processed { get; private set; } = false;
        
        /// <summary>
        /// From *PROCESSED* OnsetStrengthEnvelope and GlobalTempoEstimate initializes arrays and values.
        /// </summary>
        void InitialSetup()
        {
            Scoring = new float[Envelope.Envelope.Length];
            Predestors = new int[Scoring.Length];
            IdealTempo = (int)TempoEstimate.IdealInterBeatIntervalInSamples;
        }

        /// <summary>
        /// Computes beats from discrete signal. During process computes OnsetStrengthEnvelope and GlobalTempoEstimate classes.
        /// </summary>
        /// <param name="signal">Discrete signal to derive beats from.</param>
        public DynamicBeatTracking(DiscreteSignal signal)
        {
            var envelope = new OnsetStrengthEnvelope(signal);
            envelope.ProcessEnvelope();
            var tempoEstimate = new GlobalTempoEstimate(envelope,true);
            tempoEstimate.ProcessTempoEstimation();

            Envelope = envelope;
            TempoEstimate = tempoEstimate;

            InitialSetup();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="envelope">Onset strength envelope. If not processed then will be processed.</param>
        /// <param name="tempoEstimate">Global tempo estimate. If not processed, then will be processed.</param>
        public DynamicBeatTracking(OnsetStrengthEnvelope envelope, GlobalTempoEstimate tempoEstimate)
        {
            Envelope = envelope;
            if(!Envelope.Processed) { Envelope.ProcessEnvelope(); }

            TempoEstimate = tempoEstimate;
            if (!TempoEstimate.Processed) { TempoEstimate.ProcessTempoEstimation(); }

            InitialSetup();
        }

        /// <summary>
        /// Computes beats from provided envelope and tempo estimation. 
        /// </summary>
        public void ProcessBeats()
        {
            if(Processed)
            {
                return;
            }

            ComputeScoring();
            ComputeBeats();
            ComputeTempoAndDeviation();
            Processed = true;
        }

        /// <summary>
        /// Computes average beat interval, standard deviation and Beats-Per-Minute tempo.
        /// </summary>
        void ComputeTempoAndDeviation()
        {
            BeatsProcessing.StandardInterBeatDeviation(Beats,  out var average , out var deviation);
            AverageBeatInterval = average;
            StandardInterBeatDeviation = deviation;

            OverallTempoBPM = 60*SampleRate / AverageBeatInterval;
        }

        #region Compute scoring
        /// <summary>
        /// Computes error function of interval between beats.
        /// </summary>
        /// <param name="beatInterval">Actual interval between this and last beat.</param>
        /// <param name="idealInterval">Ideal interval between beats.</param>
        /// <returns></returns>
        float ConsistencyErrorFunction(int beatInterval, int idealInterval)
        {
            return -(float)Math.Pow(Math.Log(beatInterval/ idealInterval), 2);            
        }

        /// <summary>
        /// Finds maximum previous score corrected by Error function of difference between ideal interval and actual interval. 
        /// </summary>
        /// <param name="actualTime">Time where you are calculating score.</param>
        /// <param name="lowerRangeOfSearch"></param>
        /// <param name="upperRangeOfSearch"></param>
        /// <param name="predestor">Returns index of preceding beat time that gives the best score.</param>
        /// <returns></returns>
        float FindBestPrecedingScore(int actualTime, int lowerRangeOfSearch, int upperRangeOfSearch, out int predestor)
        {
            int d = lowerRangeOfSearch;
            float max = Alpha * ConsistencyErrorFunction(actualTime - d, IdealTempo) + Scoring[d];
            predestor = d;
            for (int i = lowerRangeOfSearch; i < upperRangeOfSearch+1; i++)
            {
                var potentialCandidate = Alpha * ConsistencyErrorFunction(actualTime - i, IdealTempo) + Scoring[i];
                if (potentialCandidate>max)
                {
                    max = potentialCandidate;
                    predestor = i;
                }
            }
            return max;           
        }

        /// <summary>
        /// Computes best scoring for each time and also tracks route.
        /// </summary>
        void ComputeScoring()
        {
            for (int i = 0; i < Scoring.Length; i++)
            {                
                var bestPrecedingScore = FindBestPrecedingScore(i, LowerRange(i), UpperRange(i), out int predestor);
                //potentional condition: i<IdealTempo
                if (bestPrecedingScore<0)
                {
                    bestPrecedingScore = 0;
                    predestor = -1;
                }
                
                Scoring[i] = Envelope.Envelope[i] + bestPrecedingScore;
                Predestors[i] = predestor;
            }
        }

        /// <summary>
        /// Returns lower range of considered time-window according to growing penalty when far from estimated tempo.
        /// </summary>
        /// <param name="time"></param>
        /// <returns></returns>
        int LowerRange(int time)
        {
            return Math.Max(time - 2 * IdealTempo,0);
        }

        /// <summary>
        /// Returns upper range of considered time-window according to growing penalty when far from estimated tempo.
        /// </summary>
        /// <param name="time"></param>
        /// <returns></returns>
        int UpperRange(int time)
        {
            return Math.Max(time - IdealTempo / 2,0);
        }
        #endregion

        #region Find beat intervals

        /// <summary>
        /// Computes beats from scoring and predestors.
        /// </summary>
        void ComputeBeats()
        {
            List<int> reversedBeats = new List<int>();

            var max = Scoring.Max();
            var maxIndex = Array.IndexOf(Scoring, max);
            
            var index = maxIndex;
            while(index>0)
            {
                reversedBeats.Add(index);
                index = Predestors[index];
            }
            // beats are reversed, because we are bactracing from biggest value.
            reversedBeats.Reverse();

            var beats = reversedBeats;

            Beats = beats.ToArray();
        }

        #endregion

        /// <summary>
        /// Returns times of beats (in seconds).
        /// </summary>
        /// <returns></returns>
        public float[] TimeBeats()
        {
            var values = new float[Beats.Length];
            for (int i = 0; i < values.Length; i++)
            {
                //Beats are in samples -> to seconds divide it by sample rate.
                values[i] = Beats[i] / (float)SampleRate;
            }
            return values;
        }

    }


    #endregion

    #region beats postprocessing

    /// <summary>
    /// Static class to get information from beats.
    /// </summary>
    public static class BeatsProcessing
    {
        /// <summary>
        /// Computes average beat interval and standard deviation.
        /// </summary>
        /// <param name="beats"></param>
        /// <param name="averageBeatInterval"></param>
        /// <param name="standardDeviation"></param>
        public static void StandardInterBeatDeviation(int[] beats, out float averageBeatInterval, out float standardDeviation)
        {
            var interBeatsIntervals = new List<int>();
            for (int i = 1; i < beats.Length; i++)
            {
                interBeatsIntervals.Add(beats[i] - beats[i - 1]);
            }

            double averageInterval = ((double)interBeatsIntervals.Sum() / interBeatsIntervals.Count);

            double deviationSum = 0;
            foreach(var i in interBeatsIntervals)
            {
                deviationSum += (averageInterval - i) * (averageInterval - i);
            }
            double deviation = Math.Sqrt(deviationSum / interBeatsIntervals.Count);

            averageBeatInterval = (float)averageInterval;
            standardDeviation = (float)deviation;
        }
    }


    #endregion

}
