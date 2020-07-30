using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NWaves.Signals;

namespace FeaturesExtraction
{

    /// <summary>
    /// Computes average energy and its standard deviation from provided discrete signal.
    /// </summary>
    public class Energy
    {
        // computes average energy and its deviation according to https://pdfs.semanticscholar.org/3232/058b0f79080a4b047897efbc96cfd40bf89e.pdf


        /// <summary>
        /// Average energy of the signal.
        /// </summary>
        public float AverageEnergy { get; private set; }
        /// <summary>
        /// Standard deviation of energy of the signal.
        /// </summary>
        public float StandardDeviation { get; private set; }
        /// <summary>
        /// Indicates if computation is processed.
        /// </summary>
        public bool Processed { get; private set; } = false;

        DiscreteSignal Signal { get; set; }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="signal"></param>
        public Energy(DiscreteSignal signal)
        {
            Signal = signal;
        }

        /// <summary>
        /// Computes average energy and standard deviation.
        /// </summary>
        public void ProcessEnergy()
        {
            if (Processed)
            {
                return;
            }

            double energySum = 0;            
            foreach (var t in Signal.Samples)
            {
                energySum += t * t;                
            }

            AverageEnergy = (float) (energySum / Signal.Samples.Length);

            double deviationSum = 0;
            foreach(var t in Signal.Samples)
            {
                deviationSum += (AverageEnergy - t) * (AverageEnergy - t);
            }
            StandardDeviation = (float) Math.Sqrt(deviationSum / Signal.Samples.Length);

            Processed = true;
        }
    }
}
