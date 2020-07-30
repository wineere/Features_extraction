using CommandLine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FeaturesExtraction
{
    /// <summary>
    /// Static class to simplify handling arguments from command line.
    /// </summary>
    public static class CommandLineParsing
    {
        /// <summary>
        /// Method to parse and execute command-line arguments.
        /// </summary>
        /// <param name="args"></param>
        public static void Parser(string[] args)
        {
            CommandLine.Parser.Default.ParseArguments<Options>(args)
                .WithParsed(RunOptions)
                .WithNotParsed(HandleParseError);
        }

        /// <summary>
        /// Class to describe possible parameters and commands in console.
        /// </summary>
        class Options
        {
            [Option('r', "read", Required = true,
                HelpText = "Input file to be processed.")]
            public string InputFile { get; set; }

            [Option('f', "features", Default = false,
                HelpText = "Extracts features and saves them to file.")]
            public bool GetFeatures { get; set; }

            [Option("comments", Default = true,
                HelpText = "If comments should be included to features.")]
            public bool WithComments { get; set; }

            [Option('c', "addClap", Default = false,
                HelpText = "Add clapping to audio track and saves it to file.")]
            public bool AddClapping { get; set; }

            [Option('j', "json", Default = false,
                HelpText = "Extracts features and saves them to json file.")]
            public bool Json { get; set; }
        }

        /// <summary>
        /// To be called after successful parsing.
        /// </summary>
        /// <param name="opts"></param>
        static void RunOptions(Options opts)
        {
            if (opts.GetFeatures)
            {
                if (opts.Json)
                {
                    ProgramOutput.ExtractFeaturesAndSaveToJSON(opts.InputFile);
                }
                else
                {
                    ProgramOutput.ExtractFeatures(opts.InputFile, opts.WithComments);
                }
            }
            if (opts.AddClapping)
            {
                ProgramOutput.AddClapsOnRhytm(opts.InputFile);
            }
        }
        /// <summary>
        /// To be called after unsuccessful parsing.
        /// </summary>
        /// <param name="errs"></param>
        static void HandleParseError(IEnumerable<Error> errs)
        {
            Console.WriteLine("Wrong input, try again.");
        }
    }
}
