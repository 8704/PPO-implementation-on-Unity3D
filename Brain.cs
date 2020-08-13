namespace REINFORCE {
    using System;
    using System.Collections.Generic;
    using UnityEngine;
    using System.Linq;

    public class Brain : MonoBehaviour {
        [NonSerialized] Policy policy;
        [NonSerialized] float[][][] graph;
        public List<List<float>> STATE = new List<List<float>>();
        public List<float> state = new List<float>();
        public List<float> ACTION = new List<float>();
        public List<float> RAWACTION = new List<float>();
        public float REWARD;
        public int stateSize;
        public int memorySize = 1;
        public int dimensionSize = 1;
        public int actionSize;
        public int ID;
        public float accumulatedReward;
        public bool isActive = false;
        public int steps;
        public bool processBrain = false;
        public bool humanControlled = false;
        [NonSerialized] Stream myStream;
        bool episodeEnded = false;
        public void Initialize(Policy policy_) {
            policy = policy_;
            actionSize = policy.actionSize;
            stateSize = policy.stateSize;
            dimensionSize = policy.dimensions;
            memorySize = policy.memorySize;
        }
        public virtual void ResetPosition() {

        }
        public virtual void StartEpisode() {
            if (policy.isReady) {
                (myStream, graph) = policy.ClaimNewEpisode();
                //graph = Graph.AddNoise(graph, 0.01f);
                ID = 0;
                isActive = true;
            } else {
                ID = -1;
                return;
            }
            steps = 0;
            STATE.Clear();
            ACTION.Clear();
            REWARD = 0;
            accumulatedReward = 0;
            for (int a = 0; a < memorySize; a++)
                STATE.Add(Enumerable.Repeat(0f, stateSize * dimensionSize).ToList());
            ACTION = Enumerable.Repeat(0f, actionSize).ToList();
            RAWACTION = Enumerable.Repeat(0f, actionSize * 2).ToList();
        }
        public List<float> GetAction(List<float> observation) {
            if (steps != 0) {
                policy.ToStream(STATE, ACTION, RAWACTION, REWARD, ID, myStream);
            }
            steps++;
            REWARD = 0;
            STATE.Add(observation);
            CalculateStateTrajectory();
            STATE.RemoveAt(0);
            float[] ACTION_;
            float[] RAWACTION_;
            //(ACTION_, RAWACTION_) = policy.GetAction(STATE);
            (ACTION_, RAWACTION_) = Graph.ActorOutput(graph, Flatten(STATE).ToArray(), policy.entropy);
            ACTION = ACTION_.ToList();
            RAWACTION = RAWACTION_.ToList();
            List<float> returnAction = new List<float>();
            for (int a = 0; a < ACTION.Count; a++) {
                returnAction.Add(Mathf.Clamp(ACTION[a], -1f, 1f));
            }
            return returnAction;
        }
        public void CalculateStateTrajectory() {
            int d = 0;
            for (int a = 0; a < dimensionSize - 1; a++) {
                float[] change = new float[stateSize];
                for (int b = 0; b < stateSize; b++, d++) {
                    change[b] = (STATE[memorySize][d] - STATE[memorySize - 1][d]) * 0.5f;
                }
                STATE[memorySize].AddRange(change);
            }
        }
        List<float> Flatten(List<List<float>> state) {
            List<float> returnState = new List<float>();
            for (int a = 0; a < state.Count; a++) {
                returnState.AddRange(state[a]);
            }
            return returnState;
        }
        public void AddReward(float reward) {
            REWARD += reward;
            accumulatedReward += reward;
        }
        public virtual void EndEpisode(float terminalReward) {
            if (isActive) {
                REWARD += terminalReward;
                accumulatedReward += terminalReward;
                policy.ToStream(STATE, ACTION, RAWACTION, REWARD, ID, myStream);
                policy.EndStream(0, accumulatedReward, myStream);
                if (processBrain) {
                    //policy.GetValueEstimateAndGae(policy.stream[ID]);
                    //policy.PrintStream(policy.stream[ID]);
                    processBrain = false;
                }
                if (humanControlled) {
                    policy.stream.RemoveAt(ID);
                }
                ID = -1;
                isActive = false;
            }
        }
    }

}
