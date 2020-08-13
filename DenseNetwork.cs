using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading.Tasks;
using System.Threading;
using REINFORCE;
using System;
using System.Linq;
using Sirenix.OdinInspector;
using System.Collections.Concurrent;

[System.Serializable]
public class DenseNetwork {
    [System.Serializable]
    public struct LayerArchitecture {
        public int inputSize;
        public int units;
        public LayerArchitecture(int inputSize_, int units_) {
            inputSize = inputSize_;
            units = units_;
        }
    }
    public enum NetworkType { Policy, Critic, FeatureEncoder, ActionPredictor, Explorer };
    public List<LayerArchitecture> architecture;
    public List<DenseLayer> layers;
    public Policy policy;
    public float[] TRUTH;
    public int outputLength;
    public int totalSamples;
    public float criticClipRange;

    [NonSerialized] public float[][][] gradients;

    #region Initialize
    public virtual void Initialize(Policy policy_, NetworkType networkType) {
        layers = new List<DenseLayer>();
        policy = policy_;
        switch (networkType) {
            case NetworkType.Policy:
                ReshapeArchiture(policy.stateSize * policy.memorySize * policy.dimensions, policy.actionSize*2);
                break;
            case NetworkType.Critic:
                ReshapeArchiture(policy.stateSize * policy.memorySize * policy.dimensions, 1);
                break;
            case NetworkType.FeatureEncoder:
                ReshapeArchiture(policy.stateSize * policy.memorySize * policy.dimensions, architecture[architecture.Count - 1].units);
                break;
            case NetworkType.ActionPredictor:
                ReshapeArchiture(policy.featureEncoderArchitecture[policy.featureEncoderArchitecture.Count - 1].units * 2, policy.actionSize * 2);
                break;
            case NetworkType.Explorer:
                ReshapeArchiture(policy.featureEncoderArchitecture[policy.featureEncoderArchitecture.Count - 1].units + policy.actionSize * 2, 
                    policy.featureEncoderArchitecture[policy.featureEncoderArchitecture.Count - 1].units);
                break;
        }
        for (int a = 0; a < architecture.Count; a++) {
            layers.Add(new DenseLayer(architecture[a].units, architecture[a].inputSize));
        }
        gradients = CreateGradientsBank();
        outputLength = architecture[layers.Count - 1].units;
    }
    public void ReshapeArchiture(int inputSize, int outputSize) {
        int previousLayersUnits = inputSize;
        List<int> inputSizes = new List<int>();
        for (int a = 0; a < architecture.Count; a++) {
            inputSizes.Add(1 + previousLayersUnits);
            previousLayersUnits = architecture[a].units;
        }
        for (int a = 0; a < architecture.Count; a++) {
            architecture[a] = new LayerArchitecture(inputSizes[a], architecture[a].units);
        }
        architecture[architecture.Count - 1] = new LayerArchitecture(inputSizes.Last(), outputSize);
    }
    public float[][][] CreateGradientsBank() {
        float[][][]  gradients = new float[layers.Count][][];
        for (int a = 0; a < layers.Count; a++) {
            gradients[a] = new float[layers[a].nubW.Length][];
            for (int b = 0; b < layers[a].nubW.Length; b++) {
                gradients[a][b] = new float[layers[a].nubW[b].Length];
                for (int c = 0; c < layers[a].nubW[b].Length; c++) {
                    gradients[a][b][c] = new float();
                }
            }
        }
        return gradients;
    }

    public float[][] CreateLayersInputs() {
        float[][] layersInputs = new float[layers.Count][];
        for (int a = 0; a < layers.Count; a++) {
            layersInputs[a] = new float[layers[a].nubW[0].Length - 1];
        }
        return layersInputs;
    }
    public float[][] CreateSwishBank() {
        float[][] swishBank = new float[layers.Count][];
        for (int a = 0; a < layers.Count; a++) {
            swishBank[a] = new float[layers[a].nubW.Length];
        }
        for (int a = 0; a < swishBank[swishBank.Length - 1].Length; a++) {
            swishBank[swishBank.Length - 1][a] = 1f;
        }
        return swishBank;
    }
    public void UpdateWeights(int batchSize) {
        totalSamples+= batchSize;
        for (int a = 0; a < layers.Count; a++) {
            for (int b = 0; b < layers[a].nubW.Length; b++) {
                for (int c = 0; c < layers[a].nubW[b].Length; c++) {
                    float delta = gradients[a][b][c] / batchSize;
                    layers[a].nubRMS[b][c] = 0.999f * layers[a].nubRMS[b][c] + 0.001f * (delta * delta);
                    layers[a].nubMomentum[b][c] = 0.9f * layers[a].nubMomentum[b][c] + 0.1f * (delta);
                    layers[a].nubW[b][c] -= layers[a].lr * (layers[a].nubMomentum[b][c] / (Mathf.Sqrt(layers[a].nubRMS[b][c]) + 0.000000001f));
                    gradients[a][b][c] = 0;
                }
            }
        }
    }
    #endregion
    #region Loading
    public void CopyNetwork(DenseNetwork networkToCopy) {
        architecture = networkToCopy.architecture;
        outputLength = architecture[architecture.Count - 1].units;
        layers = new List<DenseLayer>();
        for (int a = 0; a < networkToCopy.layers.Count; a++) {
            layers.Add(new DenseLayer(networkToCopy.layers[a].totalNodes, networkToCopy.layers[a].totalNubs));
        }
        float[][][] graph = networkToCopy.GetGraph();
        LoadGraph(graph);
    }
    public void CopyWeights(DenseNetwork networkToCopy, float interpolation = 1f) {
        float[][][] graph = networkToCopy.GetGraph();
        for (int a = 0; a < graph.Length; a++) {
            for (int b = 0; b < graph[a].Length; b++) {
                for (int c = 0; c < graph[a][b].Length; c++) {
                    layers[a].nubW[b][c] = (interpolation * graph[a][b][c]) + ((1f - interpolation) * layers[a].nubW[b][c]);
                }
            }
        }
    }
    public float[][][] GetGraph() {
        float[][][] graph = new float[layers.Count][][];
        int layersCount = layers.Count;
        for (int a = 0; a < layersCount; a++) {
            int nubWLength = layers[a].nubW.Length;
            int nubWLength2 = layers[a].nubW[0].Length;
            graph[a] = new float[nubWLength][];
            for (int b = 0; b < nubWLength; b++) {
                graph[a][b] = new float[nubWLength2];
                for (int c = 0; c < nubWLength2; c++) {
                    graph[a][b][c] = layers[a].nubW[b][c];
                }
            }
        }
        return graph;
    }
    public (float[][][], float[][][], float[][][]) BackupGraph() {
        float[][][] graph = new float[layers.Count][][];
        float[][][] rms = new float[layers.Count][][];
        float[][][] momentum = new float[layers.Count][][];
        for (int a = 0; a < layers.Count; a++) {
            graph[a] = new float[layers[a].nubW.Length][];
            rms[a] = new float[layers[a].nubW.Length][];
            momentum[a] = new float[layers[a].nubW.Length][];
            for (int b = 0; b < layers[a].nubW.Length; b++) {
                graph[a][b] = new float[layers[a].nubW[b].Length];
                rms[a][b] = new float[layers[a].nubW[b].Length];
                momentum[a][b] = new float[layers[a].nubW[b].Length];
                for (int c = 0; c < layers[a].nubW[b].Length; c++) {
                    graph[a][b][c] = layers[a].nubW[b][c];
                    rms[a][b][c] = layers[a].nubRMS[b][c];
                    momentum[a][b][c] = layers[a].nubMomentum[b][c];
                }
            }
        }
        return (graph, rms, momentum);
    }
    public void LoadGraph(float[][][] graph) {
        for (int a = 0; a < graph.Length; a++) {
            for (int b = 0; b < graph[a].Length; b++) {
                for (int c = 0; c < graph[a][b].Length; c++) {
                    layers[a].nubW[b][c] = graph[a][b][c];
                }
            }
        }
    }
    public void LoadGraph(float[][][] graph, float[][][] rms, float[][][] momentum) {
        for (int a = 0; a < graph.Length; a++) {
            layers[a].samples--;
            for (int b = 0; b < graph[a].Length; b++) {
                for (int c = 0; c < graph[a][b].Length; c++) {
                    layers[a].nubW[b][c] = graph[a][b][c];
                    layers[a].nubRMS[b][c] = rms[a][b][c];
                    layers[a].nubMomentum[b][c] = momentum[a][b][c];
                }
            }
        }
    }
    public void LoadGraph(DenseNetwork network_) {
        float[][][] graph = network_.GetGraph();
        for (int a = 0; a < graph.Length; a++) {
            for (int b = 0; b < graph[a].Length; b++) {
                for (int c = 0; c < graph[a][b].Length; c++) {
                    layers[a].nubW[b][c] = graph[a][b][c];
                }
            }
        }
    }
    #endregion
    #region Training
    public float[] CalculateHiddenOutputs(float[] generatedInputs_) {
        float[] generatedInputs = generatedInputs_;
        for (int a = 0; a < layers.Count - 1; a++) {
            layers[a].Concatenate(generatedInputs);
            layers[a].ReduceSwishLayer();
            generatedInputs = layers[a].output;
        }
        return generatedInputs;
    }
    public void CalculateActorOutput(List<float> state) {
        layers[layers.Count - 1].Concatenate(CalculateHiddenOutputs(state.ToArray()));
        layers[layers.Count - 1].ReduceActorLayer();
    }
    public float CalculateCriticOutput(List<float> state) {
        layers[layers.Count - 1].Concatenate(CalculateHiddenOutputs(state.ToArray()));
        layers[layers.Count - 1].ReduceCriticLayer();
        return layers[layers.Count - 1].output[0];
    }
    public float[] CalculateFeatureEncoderOutput (List<float> state) {
        layers[layers.Count - 1].Concatenate(CalculateHiddenOutputs(state.ToArray()));
        return layers[layers.Count - 1].ReduceFeatureEncoderLayer();
    }
    public void CalculateActionPredictorOutput(float[] feature0, float[] feature1) {
        List<float> features = new List<float>(feature0);
        features.AddRange(feature1);
        layers[layers.Count - 1].Concatenate(CalculateHiddenOutputs(features.ToArray()));
        layers[layers.Count - 1].ReduceLinearLayer();
    }
    public void CalculateExplorerOutput(float[] feature0, float[] aRaw) {
        List<float> features = new List<float>(feature0);
        features.AddRange(aRaw);
        layers[layers.Count - 1].Concatenate(CalculateHiddenOutputs(features.ToArray()));
        layers[layers.Count - 1].ReduceLinearLayer();
    }
    public void BackpropogateActor(float[] actualAction, float[] advantage) {
        layers[layers.Count - 1].DescentActor(actualAction, advantage);
        BackpropogateAdamHidden(layers[layers.Count - 1].nubDelta);
    }
    public void BackpropogateCritic(float actualStatevalue, float clippedStateValue) {
        layers[layers.Count - 1].DescentCritic(actualStatevalue, clippedStateValue);
        BackpropogateAdamHidden(layers[layers.Count - 1].nubDelta);
    }
    public (float[], float[]) BackpropogateActionPredictor(float[] rawActions) {
        layers[layers.Count - 1].DescentActionPredictor(rawActions);
        BackpropogateAdamHidden(layers[layers.Count - 1].nubDelta, false);
        float[] feature1Delta = new float[layers[0].nubDelta.Length / 2];
        float[] feature2Delta = new float[layers[0].nubDelta.Length / 2];
        for(int a = 0; a < feature1Delta.Length; a++) {
            feature1Delta[a] = layers[0].nubDelta[a];
            feature2Delta[a] = layers[0].nubDelta[a + feature1Delta.Length];
        }
        return (feature1Delta, feature2Delta);
    }
    public void BackpropogateFeatureEncoder(float[] featureDeltas) {
        layers[layers.Count - 1].DescentFeatureEncoder(featureDeltas);
        BackpropogateAdamHidden(layers[layers.Count - 1].nubDelta);
    }
    public float BackpropogateExplorer(float[] feature1) {
        float explorationError = layers[layers.Count - 1].DescentExplorer(feature1);
        BackpropogateAdamHidden(layers[layers.Count - 1].nubDelta);
        return explorationError;
    }
    public void BackpropogateAdamHidden(float[] deltaPlus, bool IgnoreFirstLayerDelta = true) {
        for (int a = layers.Count - 2; a >= 0; a--) {
            layers[a].DescentAdamRelu(deltaPlus, a == 0 ? IgnoreFirstLayerDelta : false);
            deltaPlus = layers[a].nubDelta;
        }
    }
    [NonSerialized]public List<float> rewards, gaes, tdAdvantages, values, valueEstimates, valueTarget;
    [Button]
    void SampleRewardsGaeAdvantage() {
        int id = Stats.RandomInt(0, 10000);
        Stats.ArrayToFile(id + "rewards", rewards);
        Stats.ArrayToFile(id + "GAE", gaes);
        Stats.ArrayToFile(id + "tdAdvantages", tdAdvantages);
        Stats.ArrayToFile(id + "values", values);
        Stats.ArrayToFile(id + "valueEstimates", valueEstimates);
        Stats.ArrayToFile(id + "valueTarget", valueTarget);
    }
    public void Benchmark(Stream stream) {
        rewards = stream.r;
        gaes = stream.GAE.ToList();
        tdAdvantages = stream.advantageEstimates.ToList();
        valueEstimates = stream.criticValueEstimate.ToList();
        valueTarget = stream.valueTarget.ToList();
    }

    public float advantage;
    public float[] ratio, oldProbs, intermediateProbs;
    public List<float> a0, oldOutput;
    public void TrainActorPPOSafe(Stream stream) {
        TRUTH = new float[outputLength];
        int index = Stats.RandomInt(0, stream.s.Count - 1);
        List<float> s0 = stream.Flatten(stream.s[index]);
        Benchmark(stream);
        a0 = stream.a[index];
        oldOutput = stream.aRaw[index];
        advantage = stream.GAE[index];
        oldProbs = GetProbabilities(a0, oldOutput.ToArray());
        totalSamples++;
        CalculateActorOutput(s0);
        intermediateProbs = GetProbabilities(a0);
        ratio = new float[intermediateProbs.Length];
        for (int a = 0; a < intermediateProbs.Length; a++) {
            ratio[a] = Mathf.Exp(intermediateProbs[a] - oldProbs[a]);
        }
        for (int a = 0, b = 0; a < intermediateProbs.Length; a++, b += 2) {
            if (advantage >= 0) {
                if (ratio[a] > 1.2f) {
                    TRUTH[b] = 0f; 
                    TRUTH[b + 1] = 0f; 
                } else {
                    TRUTH[b] = advantage * ratio[a] * ((a0[a] - layers[layers.Count - 1].output[b])/ (layers[layers.Count - 1].output[b+1] * layers[layers.Count - 1].output[b+1]));
                    TRUTH[b + 1] = 0f;// advantage * ratio[a] * (Stats.Squared(a0[a] - layers[layers.Count - 1].output[b]) / (layers[layers.Count - 1].output[b + 1] * layers[layers.Count - 1].output[b + 1] * layers[layers.Count - 1].output[b + 1])); //0.35f
                }
            }
            if (advantage < 0) {
                if (ratio[a] < 0.8f) {
                    TRUTH[b] = 0f; 
                    TRUTH[b + 1] = 0f; 
                } else {
                    TRUTH[b] = advantage * ratio[a] * ((a0[a] - layers[layers.Count - 1].output[b]) / (layers[layers.Count - 1].output[b + 1] * layers[layers.Count - 1].output[b + 1]));
                    TRUTH[b + 1] = 0f;// advantage * ratio[a] * (Stats.Squared(a0[a] - layers[layers.Count - 1].output[b]) / (layers[layers.Count - 1].output[b + 1] * layers[layers.Count - 1].output[b + 1] * layers[layers.Count - 1].output[b + 1])); // 0.35f
                }
            }
        }
        BackpropogateActor(a0.ToArray(), TRUTH);
    }
    public void TrainCritic(Stream stream) {
        int index = Stats.RandomInt(0, stream.s.Count - 1);
        List<float> s0 = stream.Flatten(stream.s[index]);
        float valueTarget = stream.valueTarget[index];
        float clippedValueTarget = stream.criticValueEstimate[index] + Mathf.Clamp((stream.valueTarget[index] - stream.criticValueEstimate[index]), -criticClipRange, criticClipRange);
        CalculateCriticOutput(s0);
        BackpropogateCritic(valueTarget, clippedValueTarget);
        totalSamples++;
    }
    float[] GetProbabilities(List<float> a0) {
        float[] probabilities = new float[a0.Count];
        for (int a = 0, b = 0; b < layers[layers.Count - 1].output.Length; a++, b += 2) {
            probabilities[a] = Stats.LogProb(a0[a], layers[layers.Count - 1].output[b], layers[layers.Count - 1].output[b + 1]);
        }
        return probabilities;
    }
    float[] GetProbabilities(List<float> a0, float[] compareTo) {
        float[] probabilities = new float[a0.Count];
        for (int a = 0, b = 0; b < compareTo.Length; a++, b += 2) {
            probabilities[a] = Stats.LogProb(a0[a], compareTo[b], compareTo[b + 1]);
        }
        return probabilities;
    }
    #endregion
}

//What do I need for getting gradients?
// List of nub Gradients per Weights

// Unique to each thread:
//  lastInputs
//  coldput
//  output
//  nubDelta (which becomes previous layer's derivation)

[System.Serializable]
public class DenseLayer {
    [NonSerialized] public float[][] nubW, nubRMS, nubMomentum;
    public float lr, entropy;
    public int totalNodes, totalNubs, samples;
    public float[] nubDelta, derivation, coldput, output, lastInputs;

    public DenseLayer(int nodesAmount, int nubsAmount) {
        totalNodes = nodesAmount; totalNubs = nubsAmount; lastInputs = new float[nodesAmount]; derivation = new float[nodesAmount];
        coldput = new float[nodesAmount]; output = new float[nodesAmount]; nubW = new float[nodesAmount][];
        for (int a = 0; a < nodesAmount; a++) {
            nubW[a] = new float[nubsAmount];
            for (int b = 0; b < nubsAmount; b++)
                nubW[a][b] = Stats.RANDOM() * Mathf.Pow(nubsAmount, -0.5f);
        }
        nubRMS = new float[nodesAmount][];
        for (int a = 0; a < nodesAmount; a++) {
            nubRMS[a] = new float[nubsAmount];
        }
        nubMomentum = new float[nodesAmount][];
        for (int a = 0; a < nodesAmount; a++) {
            nubMomentum[a] = new float[nubsAmount];
        }
        nubDelta = new float[nubsAmount - 1];
    }
    //-------------------------------------------------------------- DENSE LAYER CALCULATE OUTPUT
    #region CalculateOutputs
    public void Concatenate(float[] inputs) {
        List<float> allInputs = new List<float>() { 1 };
        allInputs.AddRange(new List<float>(inputs));
        lastInputs = allInputs.ToArray();
    }
    public void ReduceSwishLayer() {
        (coldput, output) = Graph.ReduceSwishLayer(nubW, lastInputs);
    }
    public void ReduceActorLayer() {
        coldput = Graph.UnsafeReduceLayer(nubW, lastInputs);
        for (int a = 0; a < output.Length; a++) {
            if (a % 2 == 0)
                output[a] = ((2f / (1f + Mathf.Exp(-2f * coldput[a]))) - 1f);//Mul is Tanh()
            else
                output[a] = entropy; // Mathf.Exp((2f / (1f + Mathf.Exp(-2f * coldput[a])) - 1f) - 1f); //entropy; // var is Exp(Tanh()-1) 0.35f
        }
    }
    public void ReduceCriticLayer() {
        coldput = Graph.UnsafeReduceLayer(nubW, lastInputs);
        for (int a = 0; a < output.Length; a++) {
            output[a] = coldput[a];
        }
    }
    public float[] ReduceFeatureEncoderLayer() {
        coldput = Graph.UnsafeReduceLayer(nubW, lastInputs);
        for (int a = 0; a < output.Length; a++) {
            output[a] = coldput[a];
        }
        return output;
    }
    public void ReduceLinearLayer() {
        coldput = Graph.UnsafeReduceLayer(nubW, lastInputs);
        for (int a = 0; a < output.Length; a++) {
            output[a] = coldput[a];
        }
    }
    #endregion
    //-------------------------------------------------------------- DENSE LAYER BACKPROPOGATION
    #region Backpropogate
    public void DescentActor(float[] actualAction, float[] advantage) {
        samples++;
        DeriveActor(actualAction, advantage);
        GetAllDeltas();
        AdamDescent();
    }
    public void DescentCritic(float actualStateValue, float clippedStateValue) {
        samples++;
        DeriveCritic(actualStateValue, clippedStateValue);
        GetAllDeltas();
        AdamDescent();
    }
    public void DescentActionPredictor(float[] rawActions) {
        samples++;
        DeriveActionPredictor(rawActions);
        GetAllDeltas();
        AdamDescent();
    }
    public void DescentFeatureEncoder(float[] featureDeltas) {
        samples++;
        DeriveFeatureEncoders(featureDeltas);
        GetAllDeltas();
        AdamDescent();
    }
    public float DescentExplorer(float[] feature1) {
        samples++;
        float explorationError = DeriveExplorer(feature1);
        GetAllDeltas();
        AdamDescent();
        return explorationError;
    }
    public void DeriveActor(float[] actualActions, float[] advantage) {
        float [] actual = new float[actualActions.Length];
        for (int a = 0, b = 0; a < totalNodes; a++) {
            actual[b] = actualActions[b];
            if (a % 2 == 0) {
                derivation[a] = advantage[a] * ((4f * Mathf.Exp(-2f * coldput[a])) / Stats.Squared(1f + Mathf.Exp(-2f * coldput[a])));
            } else {
                derivation[a] = 0f;// advantage[a] * ((4f * Mathf.Exp(-2f * coldput[a])) / Stats.Squared(1f + Mathf.Exp(-2f * coldput[a])));//advantage[a] * stdvDerive * stdvTanhDerive * output[a], -3f, 3f); 0.35f
                b++;
            }
        }
    }
    public void DeriveCritic(float actualStateValue_, float clippedStateValue_) {
        float actualStateValue = actualStateValue_;
        float clippedStateValue = clippedStateValue_;
        float vf_losses1 = (output[0] - actualStateValue) * (output[0] - actualStateValue);
        float vf_losses2 = (output[0] - clippedStateValue) * (output[0] - clippedStateValue);
        if (vf_losses1 >= vf_losses2) {
            derivation[0] = (output[0] - actualStateValue);
        } else {
            derivation[0] = 0f;
        }
    }
    public void DeriveFeatureEncoders(float[] featureGradients) {
        derivation = featureGradients;
    }
    public void DeriveActionPredictor(float[] rawActions) {
        for (int a = 0; a < rawActions.Length; a++) {
            derivation[a] = output[a] - rawActions[a];
        }
    }
    public float DeriveExplorer(float[] feature1) {
        for (int a = 0; a < feature1.Length; a++) {
            derivation[a] = output[a] - feature1[a];
        }
        float averageAbsoluteDerivation = 0;
        for (int a = 0; a < derivation.Length; a++) {
            averageAbsoluteDerivation += Mathf.Abs(derivation[a]);
        }
        return averageAbsoluteDerivation / (float)derivation.Length;
    }
    public void DescentAdamRelu(float[] nubDeltaPlus, bool isFirstLayer) {
        samples++;
        DeriveSwish(nubDeltaPlus);
        if (!isFirstLayer)
            GetAllDeltas();
        AdamDescent();
    }
    public void DeriveSwish(float[] nubDeltaPlus) {
        for (int a = 0; a < totalNodes; a++) {
            float negativeExpColdput = Mathf.Exp(-coldput[a]);
            derivation[a] = nubDeltaPlus[a] * ((1f + negativeExpColdput + coldput[a] * negativeExpColdput)/Stats.Squared(1f+ negativeExpColdput));
        }
    }
    public void GetAllDeltas() {
        nubDelta = new float[totalNubs - 1];
        for (int a = 0; a < totalNodes; a++) {
            GetDeltas(a, derivation[a]);
        }
    }
    public void GetDeltas(int nodeIndex, float derivation) {
        for (int a = 1, b = 0; a < totalNubs; a++, b++) {
            nubDelta[b] += nubW[nodeIndex][a] * derivation;
        }
        /*
        unsafe {
            fixed (float* w = &nubW[nodeIndex][0], d_ = &nubDelta[0])
                for (int a = 1, b = 0; a < totalNubs; a++, b++) {
                    d_[b] += w[a] * derivation;
                }
        }*/
    }
    public void AdamDescent() {
        if(samples < 10000) {
            for(int a = 0; a < totalNodes; a++) {
                NubsInitialAdamDescent(a, derivation[a]);
            }
        } else {
            for (int a = 0; a < totalNodes; a++) {
                NubsAdamDescent(a, derivation[a]);
            }
        }
    }
    public void NubsInitialAdamDescent(int nodeIndex, float derivation) {
        for(int a = 0; a < totalNubs; a++) {
            float delta = derivation * lastInputs[a];
            nubRMS[nodeIndex][a] = 0.999f * nubRMS[nodeIndex][a] + 0.001f * (delta * delta);
            nubMomentum[nodeIndex][a] = 0.9f * nubMomentum[nodeIndex][a] + 0.1f * (delta);
            nubW[nodeIndex][a] -= lr * delta;
        }
    }
    public void NubsAdamDescent(int nodeIndex, float derivation) {
        for (int a = 0; a < totalNubs; a++) {
            float delta = derivation * lastInputs[a];
            nubRMS[nodeIndex][a] = 0.999f * nubRMS[nodeIndex][a] + 0.001f * (delta * delta);
            nubMomentum[nodeIndex][a] = 0.9f * nubMomentum[nodeIndex][a] + 0.1f * (delta);
            nubW[nodeIndex][a] -= lr * (nubMomentum[nodeIndex][a] / (Mathf.Sqrt(nubRMS[nodeIndex][a]) + 0.000000001f));
        }
    }
    #endregion
    [Button]
    void SampleColdputs() {
        List<List<float>> sampledColdput = new List<List<float>>();
        for (int a = 0; a < nubW.Length; a++) {
            List<float> nodeColdput = new List<float>();
            for (int b = 0; b < nubW[a].Length; b++) {
                nodeColdput.Add(nubW[a][b] * lastInputs[b]);
            }
            sampledColdput.Add(nodeColdput);
        }
        Stats.ArrayToFile("Coldputs", sampledColdput);
    }
    [Button]
    void SampleWeightsAndInputs() {
        List<List<float>> sampledColdput = new List<List<float>>();
        for (int a = 0; a < nubW.Length; a++) {
            List<float> nodeColdput = new List<float>();
            for (int b = 0; b < nubW[a].Length; b++) {
                nodeColdput.Add(nubW[a][b]);
                nodeColdput.Add(lastInputs[b]);
            }
            sampledColdput.Add(nodeColdput);
        }
        Stats.ArrayToFile("WeightsAndInputs", sampledColdput);
    }
}


