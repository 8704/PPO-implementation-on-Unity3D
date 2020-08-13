namespace REINFORCE {
    using Sirenix.OdinInspector;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading;
    using System.Threading.Tasks;
    using UnityEngine;

    public class Policy : MonoBehaviour {
        public string name;
        public int stateSize;
        public int memorySize = 5;
        public int dimensions = 1;
        public int actionSize;

        public List<DenseNetwork.LayerArchitecture> actorArchitecture;
        public List<DenseNetwork.LayerArchitecture> criticArchitecture;
        public List<DenseNetwork.LayerArchitecture> featureEncoderArchitecture;
        public List<DenseNetwork.LayerArchitecture> actionPredictorArchitecture;
        public List<DenseNetwork.LayerArchitecture> explorerArchitecture;
        public NetworkGameObject targetActor;
        public NetworkGameObject targetCritic;
        public NetworkGameObject featureEncoder0Network;
        public NetworkGameObject featureEncoder1Network;
        public NetworkGameObject actionPredictorNetwork;
        public NetworkGameObject explorerNetwork;

        public List<Stream> stream = new List<Stream>();
        public List<Stream> latestStream = new List<Stream>();
        public List<Brain> brain;
        public List<float> rewardHistory;
        public float entropy = 1f;
        public float averageRewards;
        public int epochs = 4;
        public float criticLr = 0.01f;
        public float actorLr = 0.0007f;
        public float criticClipRange = 0.2f;
        public int samples = 0;
        public int maxSamples = 1000;
        public int totalSamples = 0;
        public int maxSteps = 500000;
        public bool isReady = true;
        public float averageAdvantage;
        public float stdvAdvantage;
        bool targetCriticFinished = true;
        bool targetActorFinished = true;
        bool finishedReplacingStream = true;
        public int batchPerThread = 32;
        public int threadAmounts = 8;
        public int criticEpochsTrained = 0;

        public float[][][] actorGraph;
        public float[][][] criticGraph;
        public float[][][] streamActor;
        public float[][][] latestStreamActor;
        public bool useEmptyCalculations = false;

        [FoldoutGroup("Histogram")] public int histogramPeriod = 10;
        [FoldoutGroup("Histogram")] public int intervals = 10;
        [FoldoutGroup("Histogram")] public bool useHistogram = true;
        [FoldoutGroup("Histogram")] public HistogramGraph histogram;
        [FoldoutGroup("Histogram")] public float lineWidth = 1;
        [FoldoutGroup("Histogram")] [Range(1, 100)] public int horizontalZoom = 8;
        [FoldoutGroup("Histogram")] [Range(1, 20f)] public int verticalZoom = 4;
        [FoldoutGroup("Histogram")] public float horizontalScroll = 0;
        [FoldoutGroup("Histogram")] [Range(0f, 1f)] public float verticalScroll = 0;

        [Range(0f, 1f)] public float discount = 0.98f;
        [Range(0f, 1f)] public float bootstrapRatio = 0.2f;
        [Range(0f, 1f)] public float gaeRatio = 0.92f;
        [Range(0f, 1f)] public float explorationRatio = 0.1f;


        public void Awake() {
            Initialize();
        }
        public void Initialize() {
            targetActor.network.architecture = actorArchitecture;
            targetCritic.network.architecture = criticArchitecture;
            featureEncoder0Network.network.architecture = featureEncoderArchitecture;
            featureEncoder1Network.network.architecture = featureEncoderArchitecture;
            actionPredictorNetwork.network.architecture = actionPredictorArchitecture;
            explorerNetwork.network.architecture = explorerArchitecture;

            targetActor.network.Initialize(this, DenseNetwork.NetworkType.Policy);
            targetCritic.network.Initialize(this, DenseNetwork.NetworkType.Critic);
            featureEncoder0Network.network.Initialize(this, DenseNetwork.NetworkType.FeatureEncoder);
            featureEncoder1Network.network.Initialize(this, DenseNetwork.NetworkType.FeatureEncoder);
            actionPredictorNetwork.network.Initialize(this, DenseNetwork.NetworkType.ActionPredictor);
            explorerNetwork.network.Initialize(this, DenseNetwork.NetworkType.Explorer);

            actorGraph = targetActor.network.GetGraph();
            criticGraph = targetCritic.network.GetGraph();
            streamActor = Graph.CopyGraph(actorGraph);
            foreach (Brain brain_ in brain)
                brain_.Initialize(this);
            InitializeHistogram();
            SetLearningRate();
        }
        public void FixedUpdate() {
            if (samples > maxSamples) {
                isReady = false;
                if (CanStartTraining() && targetActorFinished && finishedReplacingStream) {
                    targetActorFinished = false;
                    Task.Factory.StartNew(TrainTargetActor);
                }
            }
            if (isReady && latestStream.Count != 0 && targetCriticFinished) {
                targetCriticFinished = false;
                Task.Factory.StartNew(TrainCriticEpoch);
            }
        }
        #region Utilities
        public bool CanStartTraining() {
            if (targetCriticFinished == false)
                return false;
            for (int a = 0; a < brain.Count; a++) {
                if (brain[a].isActive == true)
                    return false;
            }
            return true;
        }
        public (Stream, float[][][]) ClaimNewEpisode() {
            Stream returnStream = new Stream();
            stream.Add(returnStream);
            return (returnStream, actorGraph);
        }
        public void ToStream(List<List<float>> state, List<float> action, List<float> rawAction, float reward, int ID, Stream stream_) {
            stream_.Add(state, action, rawAction, reward);
            samples++;
            totalSamples++;
        }
        public void EndStream(float reward, float accumulatedReward, Stream stream_) {
            histogram.UpdateGraph(accumulatedReward);
            rewardHistory.Add(accumulatedReward);
            stream_.Add(new List<List<float>>(), new List<float>(), new List<float>(), reward);
            samples++;
            totalSamples++;
            averageRewards = rewardHistory.Average();
        }
        #endregion
        /*
        public void TrainActionPredictor(List<Stream> latestStream) {
            for (int a = 0; a < latestStream.Count; a++) {
                float[][] stateFeature0 = new float[latestStream[a].s.Count][];
                float[][] stateFeature1 = new float[latestStream[a].s.Count][];
                for (int b = 0; b < latestStream[a].s.Count - 2; b++) {
                    // Get latest stream s0
                    // Feed s0 into FeatureEncoder 0
                    stateFeature0[b] = featureEncoder0Network.network.CalculateFeatureEncoderOutput(latestStream[a].sFlat[b]);
                    // Get latest stream s1
                    // Feed s1 into FeatureEncoder 1
                    stateFeature1[b] = featureEncoder1Network.network.CalculateFeatureEncoderOutput(latestStream[a].sFlat[b + 1]);
                    // Get FeatureEncoder 0 output
                    // Get FeatureEncoder 1 output
                    // Merge them
                    // Feed Merged List into Action Predictor
                    actionPredictorNetwork.network.CalculateActionPredictorOutput(stateFeature0[b], stateFeature1[b]);
                    // Compare it to latestStream a0 and Backpropogate
                    float[] feature0Delta, feature1Delta;
                    (feature0Delta, feature1Delta) = actionPredictorNetwork.network.BackpropogateActionPredictor(latestStream[a].aRaw[b].ToArray());
                    // Get Delta and backpropogate feature encoder 0
                    // Get Delta and backpropogate feature encoder 1
                    featureEncoder0Network.network.BackpropogateFeatureEncoder(feature0Delta);
                    featureEncoder1Network.network.BackpropogateFeatureEncoder(feature1Delta);
                }
                latestStream[a].stateFeature0 = stateFeature0;
                latestStream[a].stateFeature1 = stateFeature1;
            }
        }*/
        /*
        public void TrainExplorer(List<Stream> latestStream) {
            // Take Feature0 and ARaw as input
            // Tries to Predict Feature1
            // Error of Output is the reward for Exploration
            // How to balance exploration reward with real rewards?
            //          Average them
            List<float> allExplorationReward = new List<float>();
            List<float> allRewards = new List<float>();
            for (int a = 0; a < latestStream.Count; a++) {
                float[] explorationRewards = new float[latestStream[a].r.Count];
                for (int b = 0; b < latestStream[a].stateFeature0.Length - 3; b++) {
                    explorerNetwork.network.CalculateExplorerOutput(latestStream[a].stateFeature0[b], latestStream[a].aRaw[b].ToArray());
                    explorationRewards[b] = explorerNetwork.network.BackpropogateExplorer(latestStream[a].stateFeature1[b]); //this stateFeature1 is already using next state as input
                    allRewards.Add(latestStream[a].r[b]);
                }
                latestStream[a].explorationReward = explorationRewards.ToList();
                allExplorationReward.AddRange(explorationRewards);
            }
            float averageExplorationRewards = allExplorationReward.Average();
            float stdvExplorationRewards = Stats.STDV(allExplorationReward.ToArray());
            float averageRewardsThisRound = allRewards.Average();
            for (int a = 0; a < latestStream.Count; a++) {
                for (int b = 0; b < latestStream[a].explorationReward.Count - 1; b++) {
                    latestStream[a].explorationReward[b] = ((latestStream[a].explorationReward[b] - averageExplorationRewards) / stdvExplorationRewards) * Mathf.Abs(averageRewardsThisRound);
                }
            }
        }*/
        public void TrainActionPredictor() {
            float[][][] feature0Graph = featureEncoder0Network.network.GetGraph();
            float[][][] feature1Graph = featureEncoder1Network.network.GetGraph();
            float[][][] actionPredictorGraph = actionPredictorNetwork.network.GetGraph();
            for (int a = 0; a < latestStream.Count; a++) {
                TrainActionPredictorOneStream(latestStream[a], feature0Graph, feature1Graph, actionPredictorGraph);
            }
            featureEncoder0Network.network.UpdateWeights(streamDistribution[streamDistribution.Length - 1]);
            featureEncoder1Network.network.UpdateWeights(streamDistribution[streamDistribution.Length - 1]);
            actionPredictorNetwork.network.UpdateWeights(streamDistribution[streamDistribution.Length - 1]);
        }
        public void TrainActionPredictorOneStream(Stream stream, float[][][] feature0Graph, float[][][] feature1Graph, float[][][] actionPredictorGraph) {
            float[][] stateFeature0 = new float[stream.s.Count][];
            float[][] stateFeature1 = new float[stream.s.Count][];
            Task<(float[], float[])>[] tasks = new Task<(float[], float[])>[stream.s.Count - 3];
            for (int b = 0; b < stream.s.Count - 3; b++) {
                tasks[b] = Task.Factory.StartNew(() => Graph.GetActorPredictorGradients(feature0Graph, feature1Graph, actionPredictorGraph,
                    stream.sFlat[b], stream.sFlat[b + 1], stream.aRaw[b],
                    featureEncoder0Network.network.gradients, featureEncoder1Network.network.gradients, actionPredictorNetwork.network.gradients));
                Stats.SlowMath();
            }
            for (int b = 0; b < stream.s.Count - 3; b++) {
                (stateFeature0[b], stateFeature1[b]) = tasks[b].Result;
            }
            stream.stateFeature0 = stateFeature0;
            stream.stateFeature1 = stateFeature1;

        }
        public void TrainExplorer() {
            float[][][] explorerGraph = explorerNetwork.network.GetGraph();
            for (int a = 0; a < latestStream.Count; a++) {
                TrainExplorerOneStream(latestStream[a], explorerGraph);
            }
            explorerNetwork.network.UpdateWeights(streamDistribution[streamDistribution.Length - 1]);
            NormalizeExplorationRewards();
        }
        public float averageExplorationRewards;
        public float stdvExplorationRewards;
        public float averageRewardsThisRound;
        public void NormalizeExplorationRewards() {
            List<float> allExplorationReward = new List<float>();
            List<float> allRewards = new List<float>();
            for (int a = 0; a < latestStream.Count; a++) {
                for (int b = 0; b < latestStream[a].explorationReward.Length; b++) {
                    allExplorationReward.Add(latestStream[a].explorationReward[b]);
                    allRewards.Add(latestStream[a].r[b]);
                }
            }
             averageExplorationRewards = allExplorationReward.Average();
             stdvExplorationRewards = Stats.STDV(allExplorationReward.ToArray());
             averageRewardsThisRound = allRewards.Average();
            for (int a = 0; a < latestStream.Count; a++) {
                for (int b = 0; b < latestStream[a].explorationReward.Length - 1; b++) {
                    latestStream[a].explorationReward[b] = ((latestStream[a].explorationReward[b] - averageExplorationRewards) / stdvExplorationRewards) * Mathf.Abs(averageRewardsThisRound);
                    if (averageRewardsThisRound > averageRewards) {
                        latestStream[a].explorationReward[b] *= 0.5f;
                    }
                }
            }
        }
        public void TrainExplorerOneStream(Stream stream, float[][][] explorereGraph) {
            Task<float>[] tasks = new Task<float>[stream.s.Count - 4];
            float[] explorationReward_ = new float[stream.r.Count];
            for (int b = 0; b < stream.s.Count - 4; b++) {
                tasks[b] = Task.Factory.StartNew(() => Graph.GetExplorerGradients(explorereGraph, 
                    stream.stateFeature0[b], stream.aRaw[b], stream.stateFeature1[b], explorerNetwork.network.gradients));
                Stats.SlowMath();
            }
            for (int b = 0; b < stream.s.Count - 4; b++) {
                explorationReward_[b] = tasks[b].Result;
            }
            stream.explorationReward = explorationReward_;
        }
        public void TrainCriticEpoch() {
            GetAllValuesAndGaes(latestStream);
            for (int a = 0; a < samples; a += batchPerThread * threadAmounts) {
                TrainCriticBatch();
            }
            criticEpochsTrained++;
            targetCriticFinished = true;
        }
        public void TrainCriticBatch() {
            criticGraph = targetCritic.network.GetGraph();
            Task[] tasks = new Task[threadAmounts];
            for (int b = 0; b < threadAmounts; b++) {
                tasks[b] = Task.Factory.StartNew(GetCriticGradient);;
                Stats.SlowMath();
            }
            Task.WaitAll(tasks);
            targetCritic.network.UpdateWeights(batchPerThread * threadAmounts);
        }
        public void GetCriticGradient() {
            for (int a = 0; a < batchPerThread; a++) {
                Graph.GetCriticGradient(criticGraph, latestStream[RandomStreamIndex()], targetCritic.network.gradients);
            }
        }
        public void TrainActorEpoch() {
            for (int a = 0; a < samples; a += batchPerThread * threadAmounts) {
                TrainActorBatch();
            }
        }
        public void TrainActorBatch() {
            actorGraph = targetActor.network.GetGraph();
            Task[] tasks = new Task[threadAmounts];
            for (int b = 0; b < threadAmounts; b++) {
                tasks[b] = Task.Factory.StartNew(GetActorGradient);
                Stats.SlowMath();
            }
            Task.WaitAll(tasks);
            targetActor.network.UpdateWeights(batchPerThread * threadAmounts);
        }
        public void GetActorGradient() {
            for (int a = 0; a < batchPerThread; a++) {
                Graph.GetActorGradient(actorGraph, latestStream[RandomStreamIndex()], targetActor.network.gradients);
            }
        }
        public void TrainTargetActor() {
            if (latestStream.Count != 0) {
                ThreadPool.SetMaxThreads(threadAmounts * 2, threadAmounts * 2);
                var watch = System.Diagnostics.Stopwatch.StartNew();
                for(int e = criticEpochsTrained; e < epochs; e++) {
                    TrainCriticEpoch();
                }
                GetAllValuesAndGaes(latestStream);
                NormalizeGAE(latestStream);
                for (int e = 0; e < epochs; e++) {
                    TrainActorEpoch();
                }
                watch.Stop();
                Debug.Log($"Execution Time: {watch.ElapsedMilliseconds} ms");
            }
            actorGraph = targetActor.network.GetGraph();
            ReplaceLatestStream(ReadyLatestStream(stream));
            SetLearningRate();
            samples = 0;
            criticEpochsTrained = 0;
            targetActorFinished = true;
            isReady = true;
        }
        public void GetAllValuesAndGaes(List<Stream> streams) {
            criticGraph = targetCritic.network.GetGraph();
            /*
            //Parallel.For(0, streams.Count, a => streams[a].UpdateValues(criticGraph, discount, gaeRatio));
            Task[] tasks = new Task[streams.Count];
            for (int a = 0; a < streams.Count; a++) {
                tasks[a] = Task.Factory.StartNew(() => streams[a].UpdateValues(criticGraph, discount, gaeRatio));
            }
            Task.WaitAll(tasks);
            */
            for (int a = 0; a < streams.Count; a++) {
                streams[a].UpdateValues(criticGraph, discount, gaeRatio);
            }
        }
        public void NormalizeGAE(List<Stream> stream) {
            List<float> advantages = new List<float>();
            for(int a = 0; a < stream.Count; a++) {
                for(int b=0; b < stream[a].GAE.Length; b++) {
                    advantages.Add(stream[a].GAE[b]);
                }
            }
            averageAdvantage = advantages.Average();
            stdvAdvantage = Stats.STDV(advantages.ToArray());
            for (int a = 0; a < stream.Count; a++) {
                for (int b = 0; b < stream[a].GAE.Length; b++) {
                    stream[a].GAE[b] = (stream[a].GAE[b] - averageAdvantage) / (stdvAdvantage + 0.00000001f);
                }
            }
        }
        public List<Stream> ReadyLatestStream(List<Stream> stream) {
            try {
                List<Stream> latestStream_ = new List<Stream>();
                for (int a = 0; a < stream.Count; a++) {
                    if (stream[a].r.Count > 5) {
                        latestStream_.Add(stream[a]);
                        stream[a].FlattenStates();
                    }

                }
                GetAllValuesAndGaes(latestStream_);
                return latestStream_;
            } catch (System.Exception ex) {
                Debug.Log(ex);
                return stream;
            }
        }
        public int[] streamDistribution;
        public void ReplaceLatestStream(List<Stream> latestStream_) {
            latestStream = latestStream_;
            latestStreamActor = Graph.CopyGraph(streamActor);
            streamActor = Graph.CopyGraph(actorGraph);
            stream = new List<Stream>();

            streamDistribution = new int[latestStream.Count + 1];
            int sumSamples = 0;
            for (int a = 0; a < latestStream.Count; a++) {
                sumSamples += latestStream[a].r.Count;
                streamDistribution[a] = latestStream[a].r.Count;
            }
            streamDistribution[streamDistribution.Length - 1] = sumSamples;
            TrainActionPredictor();
            TrainExplorer();
            for(int a= 0; a < latestStream.Count; a++) {
                latestStream[a].FixRewards(explorationRatio);
            }
        }
        public int RandomStreamIndex() {
            int sampleIndex = Stats.RandomInt(0, streamDistribution[streamDistribution.Length - 1]);
            int sumSamples = 0;
            for (int a = 0; a < streamDistribution.Length; a++) {
                sumSamples += streamDistribution[a];
                if (sumSamples > sampleIndex) {
                    return a;
                }
            }
            return 0;
        }
        public void SetLearningRate() {
            //entropy = Mathf.Max((1f - ((float)totalSamples / maxSteps)), 0.01f) + 0.15f;
            
            for (int a = 0, b = targetActor.network.layers.Count; a < targetActor.network.layers.Count; a++, b--) {
                if(targetActor.network.layers[a].samples != 0) {
                    targetActor.network.layers[a].lr = -Mathf.Max(actorLr * Mathf.Pow(0.5f, b - 1) * (1f - ((float)totalSamples / maxSteps)), 0.0000000001f);
                    targetActor.network.layers[a].entropy = entropy;
                } else {
                    targetActor.network.layers[a].lr = -0.0001f;
                    targetActor.network.layers[a].entropy = entropy;
                }
                
            }
            for (int a = 0, b = targetCritic.network.layers.Count; a < targetCritic.network.layers.Count; a++, b--) {
                if(targetCritic.network.layers[a].samples != 0) {
                    targetCritic.network.layers[a].lr = Mathf.Max(criticLr * Mathf.Pow(0.5f, b - 1) * (1f - ((float)totalSamples / maxSteps)), 0.0000000001f);
                } else {
                    targetCritic.network.layers[a].lr = 0.001f;
                }
            }
            for (int a = 0, b = featureEncoder0Network.network.layers.Count; a < featureEncoder0Network.network.layers.Count; a++, b--) {
                if(featureEncoder0Network.network.layers[a].samples != 0) {
                    featureEncoder0Network.network.layers[a].lr = Mathf.Pow(0.5f, b - 1) * criticLr;
                    featureEncoder1Network.network.layers[a].lr = Mathf.Pow(0.5f, b - 1) * criticLr;
                } else {
                    featureEncoder0Network.network.layers[a].lr = 0.001f;
                    featureEncoder1Network.network.layers[a].lr = 0.001f;
                }
            }
            for (int a = 0, b = actionPredictorNetwork.network.layers.Count; a < actionPredictorNetwork.network.layers.Count; a++, b--) {
                if(actionPredictorNetwork.network.layers[a].samples != 0)
                    actionPredictorNetwork.network.layers[a].lr = Mathf.Pow(0.5f, b - 1) * criticLr;
                else
                    actionPredictorNetwork.network.layers[a].lr = 0.001f;
            }
            for (int a = 0, b = explorerNetwork.network.layers.Count; a < explorerNetwork.network.layers.Count; a++, b--) {
                if(explorerNetwork.network.layers[a].samples != 0)
                    explorerNetwork.network.layers[a].lr = Mathf.Pow(0.5f, b - 1) * criticLr;
                else
                    explorerNetwork.network.layers[a].lr = 0.001f;
            }
            targetCritic.network.criticClipRange = Mathf.Max(criticClipRange * (1f - ((float)totalSamples / maxSteps)), 0.0001f);
        }
        public void InitializeHistogram() {
            GameObject trainhistogramObject = new GameObject("Histogram");
            histogram = trainhistogramObject.AddComponent<HistogramGraph>();
            histogram.LinkToNetwork(this);
        }
        public List<float> Flatten(List<List<float>> state) {
            List<float> returnState = new List<float>();
            for (int a = 0; a < state.Count; a++) {
                returnState.AddRange(state[a]);
            }
            return returnState;
        }
    }
}
