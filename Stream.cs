namespace REINFORCE {
    using System;
    using System.Collections.Generic;
    using UnityEngine;
    using System.Linq;

    //------------------------------------------------------------------------------------------ (Stream)
    //[System.Serializable]
    public class Stream {
        [NonSerialized] public List<List<List<float>>> s = new List<List<List<float>>>();
        [NonSerialized] public List<float[]> sFlat = new List<float[]>();
        [NonSerialized] public List<List<float>> a = new List<List<float>>();
        [NonSerialized] public List<List<float>> aRaw = new List<List<float>>();
        [NonSerialized] public List<float> r = new List<float>();
        [NonSerialized] public float[] criticValueEstimate;
        [NonSerialized] public float[] valueTarget;
        [NonSerialized] public float[] advantageEstimates;
        [NonSerialized] public float[] GAE;
        [NonSerialized] public float[][] stateFeature0;
        [NonSerialized] public float[][] stateFeature1;
        [NonSerialized] public float[] explorationReward;
        private readonly object myLock = new object();

        public void CopyStream(Stream stream_) {
            s = new List<List<List<float>>>(stream_.s);
            a = new List<List<float>>(stream_.a);
            aRaw = new List<List<float>>(stream_.aRaw);
            r = new List<float>(stream_.r);
        }
        public void Add(List<List<float>> state, List<float> action, List<float> rawAction, float reward) {
            s.Add(new List<List<float>>(state));
            a.Add(new List<float>(action));
            r.Add(reward);
            aRaw.Add(rawAction);
        }
        public void FixRewards(float explorationRatio) {
            for (int a = 0; a < r.Count; a++) {
                r[a] = (1f - explorationRatio) * r[a] + explorationRatio * explorationReward[a];
            }
        }
        public void FlattenStates() {
            for (int a = 0; a < s.Count; a++) {
                sFlat.Add(Flatten(s[a]).ToArray());
            }
        }
        public void UpdateValues(float[][][] criticGraph, float discountRate, float lambda) {
            try {
                GetValueEstimates(criticGraph);
                GetAdvantages(discountRate);
                GetGAEAndValueTarget(discountRate, lambda);
            } catch (System.Exception ex) {
                Debug.Log(ex);
            }
        }
        void GetValueEstimates(float[][][] criticGraph) {
            criticValueEstimate = new float[s.Count];
            for (int a = 0; a < s.Count - 1; a++) {
                criticValueEstimate[a] = Graph.CriticOutput(criticGraph, sFlat[a]);
            }
        }
        void GetAdvantages(float discountRate) {
            advantageEstimates = new float[a.Count];
            advantageEstimates[advantageEstimates.Length - 1] = r[advantageEstimates.Length - 1] - criticValueEstimate[advantageEstimates.Length - 1];
            for (int i = a.Count - 2; i >= 0; i--) {
                advantageEstimates[i] = r[i] + discountRate * criticValueEstimate[i + 1] - criticValueEstimate[i];
            }
        }
        void GetGAEAndValueTarget(float discountRate, float lambda) {
            GAE = Stats.Discount(advantageEstimates, discountRate * lambda);
            valueTarget = Stats.AddArrays(GAE, criticValueEstimate);
        }
        public List<float> Flatten(List<List<float>> state) {
            List<float> returnState = new List<float>();
            for (int a = 0; a < state.Count; a++) {
                returnState.AddRange(new List<float>(state[a]));
            }
            return returnState;
        }
    }
}
