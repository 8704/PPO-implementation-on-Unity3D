using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading.Tasks;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using UnityEngine.Rendering;
using System.Linq;
using REINFORCE;
using Sirenix.OdinInspector;
using System.Collections.Concurrent;

public static class Graph {
    public static float[] CalculateHiddenOutputs(float[][][] graph, float[] inputs) {
        int a = 0;
        //List<float> currentInputs = new List<float>() { 1 };
        //currentInputs.AddRange(inputs);
        float[] currentInputs = inputs;
        while (a < graph.Length - 1) {
            //List<float> newInputs = new List<float>() { 1 };
            //newInputs.AddRange(ReduceSwishLayer(graph[a], currentInputs).Item2);
            currentInputs = ReduceSwishLayerWithoutDerive(graph[a], currentInputs);
            a++;
        }
        return currentInputs;
    }
    public static float CriticOutput(float[][][] graph, float[] inputs) {
        float[] currentInputs = CalculateHiddenOutputs(graph, inputs);
        int currentInputsCount = currentInputs.Length;

        float sum = graph[graph.Length - 1][0][0];//BIAS
        for (int a = 0; a < currentInputsCount; a++)
            sum += graph[graph.Length - 1][0][a + 1] * currentInputs[a];
        return sum;
    }
    public static (float[], float[]) ActorOutput(float[][][] graph, float[] inputs, float entropy) {
        float[] currentInputs = CalculateHiddenOutputs(graph, inputs);
        int graphOutputLength = graph[graph.Length - 1].Length;
        int currentInputsLength = currentInputs.Length;
        float[] coldput = new float[graphOutputLength];
        for (int n = 0; n < graphOutputLength; n++) {
            float sum = graph[graph.Length - 1][n][0];//BIAS
            for (int w = 0; w < currentInputsLength; w++) {
                sum += graph[graph.Length - 1][n][w+1] * currentInputs[w];
            }
            if (n % 2 == 0)
                coldput[n] = (2f / (1f + Mathf.Exp(-2f * sum)) - 1f);
            else
                coldput[n] = entropy; //0.35f
                //coldput[n] = Mathf.Exp(((2f / (1f + Mathf.Exp(-2f * sum)) - 1f) - 1f));
        }
        List<float> output = new List<float>();
        for (int z = 0; z < graphOutputLength; z += 2)
            output.Add(coldput[z] + Stats.RANDOM() * coldput[z + 1]);
            //output.Add(Mathf.Clamp(coldput[z] + Stats.RANDOM() * coldput[z + 1], -0.95f, 0.95f));
        return (output.ToArray(), coldput.ToArray());
    }
    public static float[] UnsafeReduceLayer(float[][] nubW, float[] input) {
        float[] output = new float[nubW.Length];
        int nodeLength = nubW.Length;
        for (int b = 0; b < nodeLength; b++) {
            output[b] = nubW[b][0];  //BIAS
        }
        for (int a = 0; a < input.Length; a++) {
            if (input[a] != 0f) {
                for (int b = 0; b < nubW.Length; b++) {
                    output[b] += (nubW[b][a+1] * input[a]);
                }
            }
        }
        return output;
    }
    /*
    public static (float[], float[]) ReduceSwishLayer(float[][] nubW, float[] input) {
        float[] coldput = new float[nubW.Length];
        float[] output = new float[nubW.Length];
        int inputLength = input.Length;
        int nodeLength = nubW.Length;
        for(int b = 0; b < nodeLength; b++) {
            coldput[b] = nubW[b][0];  //BIAS
        }
        for (int a = 0; a < inputLength; a++) {
            if (input[a] != 0f) {
                for (int b = 0; b < nodeLength; b++) {
                    coldput[b] += (nubW[b][a+1] * input[a]);
                }
            }
        }
        for (int b = 0; b < nodeLength; b++) {
            output[b] = coldput[b] / (1f + Mathf.Exp(-coldput[b]));
        }
        return (coldput, output);
    }*/
    
    public static float[] ReduceSwishLayerWithoutDerive(float[][] nubW, float[] input) {
        float[] output = new float[nubW.Length];
        int inputLength = input.Length;
        int nodeLength = nubW.Length;
        for(int b = 0; b < nodeLength; b++) {
            output[b] = nubW[b][0];  //BIAS
        }
        for (int a = 0; a < inputLength; a++) {
            if (input[a] != 0f) {
                for (int b = 0; b < nodeLength; b++) {
                    output[b] += (nubW[b][a+1] * input[a]);
                }
            }
        }
        for (int b = 0; b < nodeLength; b++) {
            output[b] = output[b] / (1f + Mathf.Exp(-output[b]));
        }
        return output;
    }
    public static (float[], float[]) ReduceSwishLayer(float[][] nubW, float[] input) {
        float[] derivations = new float[nubW.Length];
        float[] output = new float[nubW.Length];
        int inputLength = input.Length;
        int nodeLength = nubW.Length;
        for (int b = 0; b < nodeLength; b++) {
            output[b] = nubW[b][0];  //BIAS
        }
        for (int a = 0; a < inputLength; a++) {
            if (input[a] != 0f) {
                for (int b = 0; b < nodeLength; b++) {
                    output[b] += (nubW[b][a + 1] * input[a]);
                }
            }
        }

        for (int b = 0; b < nodeLength; b++) {
            float negativeExpColdput = Mathf.Exp(-output[b]);
            derivations[b] = (1f + negativeExpColdput + output[b] * negativeExpColdput) / ((1f + negativeExpColdput) * (1f + negativeExpColdput));

            output[b] = output[b] / (1f + negativeExpColdput);
        }
        return (derivations, output);
    }
    public static void GetActorGradient(float[][][] weights, REINFORCE.Stream stream, float[][][] gradients) {
        try {
            int index = Stats.RandomInt(0, stream.s.Count - 1);
            List<float> a0 = stream.a[index];
            List<float> oldOutput = stream.aRaw[index];
            float advantage = stream.GAE[index];
            float[][] layersInputs;
            float[][] swishDerivatives;
            (layersInputs, swishDerivatives) = ForwardPass(weights, stream.sFlat[index]);
            int currentInputsCount = layersInputs[layersInputs.Length - 1].Length;
            float[] newOutputs = new float[weights[weights.Length - 1].Count()];
            for (int a = 0; a < weights[weights.Length - 1].Length; a++) {
                newOutputs[a] = weights[weights.Length - 1][a][0]; //BIAS
                for (int b = 0; b < currentInputsCount; b++) {
                    newOutputs[a] += weights[weights.Length - 1][a][b + 1] * layersInputs[layersInputs.Length - 1][b];
                }
                if (a % 2 == 0)
                    newOutputs[a] = (2f / (1f + Mathf.Exp(-2f * newOutputs[a])) - 1f);
                else
                    newOutputs[a] = oldOutput[a];
            }
            float[] oldProbs = GetProbabilities(a0, oldOutput.ToArray());
            float[] intermediateProbs = GetProbabilities(a0, newOutputs);
            float[] error = new float[oldOutput.Count];

            for (int a = 0, b = 0; a < intermediateProbs.Length; a++, b += 2) {
                if (advantage >= 0) {
                    if (Mathf.Exp(intermediateProbs[a] - oldProbs[a]) > 1.2f) {
                        error[b] = 0f;
                        error[b + 1] = 0f;
                    } else {
                        error[b] = advantage * Mathf.Exp(intermediateProbs[a] - oldProbs[a]) * ((a0[a] - newOutputs[b]) / (newOutputs[b + 1] * newOutputs[b + 1]));
                        error[b + 1] = 0f;// advantage * ratio[a] * (Stats.Squared(a0[a] - layers[layers.Count - 1].output[b]) / (layers[layers.Count - 1].output[b + 1] * layers[layers.Count - 1].output[b + 1] * layers[layers.Count - 1].output[b + 1])); //0.35f
                    }
                }
                if (advantage < 0) {
                    if (Mathf.Exp(intermediateProbs[a] - oldProbs[a]) < 0.8f) {
                        error[b] = 0f;
                        error[b + 1] = 0f;
                    } else {
                        error[b] = advantage * Mathf.Exp(intermediateProbs[a] - oldProbs[a]) * ((a0[a] - newOutputs[b]) / (newOutputs[b + 1] * newOutputs[b + 1]));
                        error[b + 1] = 0f;// advantage * ratio[a] * (Stats.Squared(a0[a] - layers[layers.Count - 1].output[b]) / (layers[layers.Count - 1].output[b + 1] * layers[layers.Count - 1].output[b + 1] * layers[layers.Count - 1].output[b + 1])); // 0.35f
                    }
                }
            }
            swishDerivatives[swishDerivatives.Length - 1] = Enumerable.Repeat(1f, error.Length).ToArray();
            BackwardPass(weights, error, layersInputs, swishDerivatives, gradients);

            float[] GetProbabilities(List<float> a0_, float[] compareTo) {
                float[] probabilities = new float[a0_.Count];
                for (int a = 0, b = 0; b < compareTo.Length; a++, b += 2) {
                    probabilities[a] = Stats.LogProb(a0_[a], compareTo[b], compareTo[b + 1]);
                }
                return probabilities;
            }
        } catch (System.Exception ex) {
            Debug.Log(ex);
        }
    }
    public static void GetCriticGradient(float[][][] weights, REINFORCE.Stream stream, float[][][] gradients) {
        try {
            int index = Stats.RandomInt(0, stream.s.Count - 1);
            float truth = stream.valueTarget[index];
            float clippedTruth = stream.criticValueEstimate[index] + Mathf.Clamp((stream.valueTarget[index] - stream.criticValueEstimate[index]), -0.1f, 0.1f);

            float[][] layersInputs;
            float[][] swishDerivatives;
            (layersInputs, swishDerivatives) = ForwardPass(weights, stream.sFlat[index]);
            int currentInputsCount = layersInputs[layersInputs.Length - 1].Length;
            float criticOutput = weights[weights.Length - 1][0][0];
            for (int a = 0; a < currentInputsCount; a++)
                criticOutput += weights[weights.Length - 1][0][a + 1] * layersInputs[layersInputs.Length - 1][a];
            float[] error = new float[1];
            if (Stats.Squared(criticOutput - truth) >= Stats.Squared(criticOutput - clippedTruth))
                error[0] = criticOutput - truth;
            swishDerivatives[swishDerivatives.Length - 1] = new float[] { 1f };
            BackwardPass(weights, error, layersInputs, swishDerivatives, gradients);
        } catch(System.Exception ex) {
            Debug.Log(ex);
        }
    }
    /*
    public static (List<float[]>, List<float[]>) ForwardPass(float[][][] graph, float[] inputs) {
        int a = 0;
        List<float[]> layersInputs = new List<float[]>();
        List<float[]> swishDerivatives = new List<float[]>();
        layersInputs.Add(inputs);
        while (a < graph.Length - 1) {
            (float[], float[]) coldputOutput = ReduceSwishLayer(graph[a], layersInputs[layersInputs.Count - 1]);
            layersInputs.Add(coldputOutput.Item2);
            swishDerivatives.Add(coldputOutput.Item1);
            a++;
        }
        return (layersInputs, swishDerivatives);
    }*/
    public static (float[][], float[][]) ForwardPass(float[][][] graph, float[] inputs) {
        int a = 0;
        float[][] layersInputs = new float[graph.Length][];
        float[][] swishDerivatives = new float[graph.Length][];
        layersInputs[a] = inputs;
        while (a < graph.Length - 1) {
            (float[], float[]) coldputOutput = ReduceSwishLayer(graph[a], layersInputs[a]);
            layersInputs[a+1] = coldputOutput.Item2;
            swishDerivatives[a] = coldputOutput.Item1;
            a++;
        }
        return (layersInputs, swishDerivatives);
    }
    public static void BackwardPass(float[][][] graph, float[] errors, float[][] lastInputs, float[][] swishDerivatives, float[][][] gradients) {
        for (int a = graph.Length - 1; a >= 0; a--) {
            errors = GetLayerGradients(lastInputs[a], graph[a], swishDerivatives[a], errors, gradients[a]);
        }
    }
    public static float[] BackwardPassAndGetDelta(float[][][] graph, float[] errors, float[][] lastInputs, float[][] swishDerivatives, float[][][] gradients) {
        for (int a = graph.Length - 1; a >= 0; a--) {
            errors = GetLayerGradients(lastInputs[a], graph[a], swishDerivatives[a], errors, gradients[a]);
        }
        return errors;
    }
    public static float[] GetLayerGradients(float[] lastInputs, float[][] nubW, float[] activationDerivatives, float[] nubDeltaPlus, float[][] gradients) {
        float[] nubDelta = new float[nubW[0].Length - 1];
        int nubWLength = nubW.Length;
        for (int a = 0; a < nubWLength; a++) {
            GetNodeGradients(lastInputs, nubW[a], activationDerivatives[a], nubDeltaPlus[a], nubDelta, gradients[a]);
        }
        return nubDelta;
    }
    public static void GetNodeGradients( float[] lastInputs, float[] weights, float activationDerivative, float nubDeltaPlus, float[] nubDelta, float[] gradients) { 
        float nodeDelta = activationDerivative * nubDeltaPlus;
        int weightsLength = weights.Length;
        gradients[0] += nodeDelta;  //bias
        for (int a = 1; a < weightsLength; a++) {
            gradients[a] += lastInputs[a - 1] * nodeDelta;
            nubDelta[a - 1] += weights[a] * nodeDelta;
        }
    }
    public static float[] GetSwishDerivatives(float[] coldputs) {
        float[] derivations = new float[coldputs.Length];
        for (int a = 0; a < coldputs.Length; a++) {
            float negativeExpColdput = Mathf.Exp(-coldputs[a]);
            derivations[a] = (1f + negativeExpColdput + coldputs[a] * negativeExpColdput) / Stats.Squared(1f + negativeExpColdput);
        }
        return derivations;
    }
    public static void SaveGraph(float[][][] graph, string graphName) {
        BinaryFormatter formatter = new BinaryFormatter();
        string path = Application.persistentDataPath + "/" + graphName + ".graph";
        FileStream stream = new FileStream(path, FileMode.Create);

        formatter.Serialize(stream, graph);
        stream.Close();
    }
    public static float[][][] LoadGraph(string graphName) {
        string path = Application.persistentDataPath + "/" + graphName + ".graph";
        if (File.Exists(path)) {
            BinaryFormatter formatter = new BinaryFormatter();
            FileStream stream = new FileStream(path, FileMode.Open);
            float[][][] graph = formatter.Deserialize(stream) as float[][][];
            stream.Close();
            return graph;
        } else {
            Debug.LogError("Save file not found in " + path);
            return null;
        }
    }
    public static float[][][] CopyGraph(float[][][] graph) {
        float[][][] graph_ = new float[graph.Length][][];
        for (int a = 0; a < graph.Length; a++) {
            graph_[a] = new float[graph[a].Length][];
            for (int b = 0; b < graph[a].Length; b++) {
                graph_[a][b] = new float[graph[a][b].Length];
                for (int c = 0; c < graph[a][b].Length; c++) {
                    graph_[a][b][c] = graph[a][b][c];
                }
            }
        }
        return graph_;
    }

    public static (float[], float[]) GetActorPredictorGradients(
        float[][][] feature0Weights, float[][][] feature1Weights, float[][][] actionPredictorWeights,
        float[] s0, float[] s1, List<float> aRaw, float[][][] feature0Gradients, float[][][] feature1Gradients, float[][][] actionPredictorGradients) {
        try {
            float[][] feature0layersInputs;
            float[][] feature1layersInputs;
            float[][] actionPredictorlayerInputs;
            float[][] feature0SwishDerivatives;
            float[][] feature1SwishDerivatives;
            float[][] actionPredictorSwishDerivatives;
            (feature0layersInputs, feature0SwishDerivatives) = ForwardPass(feature0Weights, s0);
            (feature1layersInputs, feature1SwishDerivatives) = ForwardPass(feature1Weights, s1);
            float[] stateFeature0 = UnsafeReduceLayer(feature0Weights[feature0Weights.Length - 1], feature0layersInputs[feature0layersInputs.Length - 1]);
            float[] stateFeature1 = UnsafeReduceLayer(feature1Weights[feature1Weights.Length - 1], feature1layersInputs[feature1layersInputs.Length - 1]);
            List<float> mergedStateFeatures = new List<float>(stateFeature0); mergedStateFeatures.AddRange(stateFeature1);
            (actionPredictorlayerInputs, actionPredictorSwishDerivatives) = ForwardPass(actionPredictorWeights, mergedStateFeatures.ToArray());
            float[] predictedActions = UnsafeReduceLayer(actionPredictorWeights[actionPredictorWeights.Length - 1], actionPredictorlayerInputs[actionPredictorlayerInputs.Length - 1]);
            float[] actionPredictorError = Stats.GetMSEErrors(predictedActions, aRaw.ToArray());

            actionPredictorSwishDerivatives[actionPredictorSwishDerivatives.Length - 1]= Enumerable.Repeat(1f, predictedActions.Length).ToArray();
            float[] featuresErrors;
            featuresErrors = BackwardPassAndGetDelta(actionPredictorWeights, actionPredictorError, actionPredictorlayerInputs, actionPredictorSwishDerivatives, actionPredictorGradients);

            float[] feature0Error = new float[stateFeature0.Length];
            float[] feature1Error = new float[stateFeature1.Length];
            for (int a = 0; a < stateFeature0.Length; a++)
                feature0Error[a] = featuresErrors[a];
            for (int a = 0; a < stateFeature1.Length; a++)
                feature1Error[a] = featuresErrors[a + stateFeature0.Length];
            feature0SwishDerivatives[feature0SwishDerivatives.Length - 1] = Enumerable.Repeat(1f, stateFeature0.Length).ToArray();
            feature1SwishDerivatives[feature1SwishDerivatives.Length - 1] = Enumerable.Repeat(1f, stateFeature1.Length).ToArray();
            BackwardPass(feature0Weights, feature0Error, feature0layersInputs, feature0SwishDerivatives, feature0Gradients);
            BackwardPass(feature1Weights, feature1Error, feature1layersInputs, feature1SwishDerivatives, feature1Gradients);

            return (stateFeature0, stateFeature1);
        } catch (System.Exception ex) {
            Debug.Log(ex);
            return (new float[0], new float[0]);
        }
    }
    public static float GetExplorerGradients(float[][][] weights, float[] feature0, List<float> aRaw, float[] feature1, float[][][] gradients) {
        try {
            float[][] layersInputs = new float[weights.Length][];
            float[][] swishDerivatives = new float[weights.Length][];
            List<float> firstInput = new List<float>(feature0);
            firstInput.AddRange(aRaw);
            (layersInputs, swishDerivatives) = ForwardPass(weights, firstInput.ToArray());
            int currentInputsCount = layersInputs[layersInputs.Length - 1].Length;
            float[] predictedStateFeature1 = UnsafeReduceLayer(weights[weights.Length - 1], layersInputs[layersInputs.Length - 1]);
            float[] featuresErrors = Stats.GetMSEErrors(predictedStateFeature1, feature1);
            swishDerivatives[swishDerivatives.Length - 1] = Enumerable.Repeat(1f, featuresErrors.Length).ToArray();
            BackwardPass(weights, featuresErrors, layersInputs, swishDerivatives, gradients);
            for(int a = 0; a < featuresErrors.Length; a++) {
                featuresErrors[a] = Mathf.Abs(featuresErrors[a]);
            }
            return featuresErrors.Average();
        } catch (System.Exception ex) {
            Debug.Log(ex);
            return 0;
        }
    }
}