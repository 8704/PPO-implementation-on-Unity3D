using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using UnityEngine;
using UnityEngine.Rendering;
using System.IO;
using System.Diagnostics;

public static class Stats {
    private static readonly System.Random random = new System.Random();
    private static readonly object syncLock = new object();
    public static int RandomInt(int min, int max) {
        lock (syncLock) { // synchronize
            return random.Next(min, max);
        }
    }
    public static float RandomFloat(float min = 0.0f, float max = 1.0f) {
        lock (syncLock) {
            float num =  (float)random.NextDouble();
            return (num * (max - min)) + min;
        }
    }
    public static int Add(int num1, int num2) {
        return num1 + num2;
    }
    public static float Squared(float number) {
        number *= number;
        return number;
    }
    public static float Sum(float[] Samples) {
        float sum = 0;
        foreach (float f in Samples) {
            sum += f;
        }
        return sum;
    }
    public static float Mean(float[] Samples) {
        float sum = Sum(Samples);
        float mean = sum / Samples.Length;
        return mean;
    }
    public static float Mean(float Sample, float Mean, int Count) {
        if (Count == 0) Count = 1;
        float difference = Sample - Mean;
        float mean = (difference / Count) + Mean;
        return mean;
    }
    public static float ErrorSquared(float Sample, float Truth) {
        float error = Sample - Truth;
        error *= error;
        return error;
    }
    public static float MSE(float[] Samples, float Truth) {
        float errorsum = 0;
        foreach (float f in Samples) {
            errorsum += ErrorSquared(f, Truth);
        }
        float mse = errorsum / Samples.Length;
        return mse;
    }
    public static float Variance(float[] Samples) {
        float Truth = Mean(Samples);
        float errorsum = 0;
        foreach (float f in Samples) {
            errorsum += ErrorSquared(f, Truth);
        }
        float mse = errorsum / Samples.Length;
        return mse;
    }
    public static float Variance_(float Sample, float Mean, float MSE, int Count) {
        float errorsquared = ErrorSquared(Sample, Mean);
        float variance = Stats.Mean(errorsquared, MSE, Count);
        return variance;
    }
    public static float STDV(float[] Samples) {
        float stdv = Mathf.Sqrt(Variance(Samples));
        return stdv;
    }
    public static float STDV_(float Sample, float Mean, float STDV, int Count) {
        return ((Mathf.Sqrt(((Sample - Mean) * (Sample - Mean)) / Count) - STDV)/ Count) + STDV;

        //float variance = Squared(STDV);
        //float stdv = Mathf.Sqrt(Variance_(Sample, Mean, variance, Count));
        //return stdv;
    }
    public static float Sigmoid(float Value) {
        float sigmoid = 1f / (1f + Mathf.Exp(-Value));
        return sigmoid;
    }
    public static float StandardError(float[] Samples) {
        float standarderror = Stats.STDV(Samples) / Mathf.Sqrt(Samples.Length);
        return standarderror;
    }
    public static float StandardError_(float Sample, float Mean, float StandardError, int Count) {
        float standarderror = Stats.STDV_(Sample, Mean, StandardError, Count) / Mathf.Sqrt(Count);
        return standarderror;
    }
    public static Vector2 ConfidenceBound(float[] Samples) {
        float upperbound = Mean(Samples) + 1.96f * StandardError(Samples);
        float lowerbound = Mean(Samples) - 1.96f * StandardError(Samples);
        return (new Vector2(lowerbound, upperbound));
    }
    public static Vector2 ConfidenceBound(float[] Samples, float ConfidenceMultiplier) {
        float upperbound = Mean(Samples) + ConfidenceMultiplier * StandardError(Samples);
        float lowerbound = Mean(Samples) - ConfidenceMultiplier * StandardError(Samples);
        return (new Vector2(lowerbound, upperbound));
    }
    public static float SmoothStop(float Value, float Power) {
        float smoothstop = 1 - Mathf.Pow((1 - Value), Power);
        return smoothstop;
    }
    public static float SmoothStop(float Value) {
        float smoothstop = 1 - Mathf.Pow((1 - Value), 2.71828f);
        return smoothstop;
    }
    public static float SmoothStart(float Value, float Power) {
        float smoothstart = Mathf.Pow(Value, Power);
        return smoothstart;
    }
    public static float SmoothStart(float Value) {
        float smoothstart = Mathf.Pow(Value, 2.71828f);
        return smoothstart;
    }

    public static float RandomGaussian() {
        float u, v;
        float s; // this is the hypotenuse squared.
        do {
            u = RandomFloat(-1f, 1f);
            v = RandomFloat(-1f, 1f);
            s = (u * u) + (v * v);
        } while (!(s != 0 && s < 1)); // keep going until s is nonzero and less than one

        // TODO allow a user to specify how many random numbers they want!
        // choose between u and v for seed (z0 vs z1)
        float seed;
        if (RandomInt(0, 2) == 0) {
            seed = u;
        } else {
            seed = v;
        }
        // create normally distributed number.
        float z = seed * Mathf.Sqrt(-2.0f * Mathf.Log(s) / s);
        return z;
    }
    public static float[] RandomFromDistribution(float[] distribution) {
        float[] mask = new float[distribution.Length];
        float random = RandomFloat(0f, 1f);
        float currentRand = 0f;

        for (int a = 0; a < distribution.Length; a++) {
            currentRand += distribution[a];
            if(currentRand >= random) {
                mask[a] = 1;
                return mask;
            }
        }
        return mask;
    }

    public static float[] GaussianBounds(float mean, float stdv, float zScore) {
        float[] bounds = new float[2];
        bounds[0] = mean - zScore * stdv;
        bounds[1] = mean + zScore * stdv;
        return bounds;
    }
    public static float RANDOM() {
        float u, v, s;
        do {
            u = RandomFloat(-1f, 1f);
            v = RandomFloat(-1f, 1f);
            s = u * u + v * v;
        } while (s == 0 || s >= 1);
        float z = Mathf.Sqrt(-2.0f * Mathf.Log(s) / s);
        if (RandomInt(0, 2) == 0)
            return u * z;
        else
            return v * z;
    }
    public static float RANDOM(float mean, float variance) {
        return mean + RANDOM() * variance;
    }
    public static float Tanh(float input) {
        return (2f / (1 + Mathf.Exp(-2f * input))) - 1f;
    }
    public static float Tanh__(float output) {
        return -0.5f * Mathf.Log(2 / (output + 1) - 1);
    }
    public static float Tanh_(float output) {
        return 1f - output * output;
    }
    public static float Softplus(float input) {
        return Mathf.Log(1+Mathf.Exp(input));
    }
    public static float Softplus_(float coldput) {
        return 1f / (1f + Mathf.Exp(-coldput));
    }
    public static float Sigmoid_(float output) {
        return output * (1 - output);
    }

    public static float GaussianLoss(float action, float mean, float variance) {
        return Mathf.Abs(action - mean) / (variance * variance);
       // return (((action - mean) * (action - mean)) / (2 * variance * variance)) - Mathf.Log(Mathf.Sqrt(2f*Mathf.PI*variance*variance));
        //return (((action - mean) * (action - mean))/ (variance * variance));
    }
    public static float Softmax_(float coldput, float sum) {
        float output = Mathf.Exp(coldput);
        float C = sum - output;
        float upper = C * output;
        float lower = Squared(sum);
        //float lower = Mathf.Exp(2 * coldput) + 2 * C * output + C * C;
        return upper / lower;
    }

    public static float RandomGaussian(float mean, float variance) {
        return mean + RandomGaussian() * variance;
    }
    public static float Mean_(float actual, float mean, float variance) {
        return (actual - mean) / (variance * variance);
    }
    public static float Mean_(float actual, float mean, float variance, float epsilon) {
        return ((actual - mean) / ((variance + epsilon) * (variance + epsilon)));
        //variance = Mathf.Clamp(variance, 0.0000001f, 1f);
        //float logSigma = Mathf.Log(variance);
        //float sigma = Mathf.Exp(logSigma);
        //return ((actual - mean) / ((sigma + epsilon) * (sigma + epsilon)));
        //return -0.5f * ((actual - mean) / ((sigma + epsilon) * (sigma + epsilon))) + (2f * logSigma + Mathf.Log(2f * Mathf.PI));
    }
    public static float Entropy_(float variance) {
        variance = Mathf.Clamp(variance, 0.0000001f, 1f);
        float logSigma = Mathf.Log(variance);
        float log2PiE = 2.837875549f; //Mathf.Log(2f * 3.14159f * 2.71828f)
        return log2PiE + 2 * logSigma;
    }

    public static float Variance_(float actual, float mean, float variance) 
    {
        float left = -1f / (2 * Squared(variance));
        float right = Squared(actual - mean) / (2f * variance * variance * variance * variance);
        return left + right;
        //return ((actual - mean) * (actual - mean) - (variance * variance)) / (variance * variance * variance);
        //return (((actual - mean) * (actual - mean)) / (variance * variance)) - 1f;
    }
    public static float MSE_(float output, float truth) {
        return output - truth;
    }
    public static float LogProb(float actual, float mean, float var) {
        return -0.5f * Stats.Squared((actual-mean)/ (var)) + 2f * Mathf.Log(var) + Mathf.Log(2f * Mathf.PI);
    }
    public static float ProbabilityDensity(float actual, float mean, float var) {
        float left = 1f / (var * 1.837877066f); //log(2pi)
        //float left = Mathf.Log(var) / Mathf.Sqrt(2f * Mathf.PI);
        float right = Mathf.Exp(-Squared(actual - mean) / (2 * var * var));
        return left * right;
    }
    public static void ArrayToFile(string name, List<float> toSave) {
        string serializedData = "";
        for (int a = 0; a < toSave.Count; a++) {
            serializedData += toSave[a];
            if(a!=toSave.Count -1)
                serializedData += "\n";
        }
        StreamWriter writer = new StreamWriter(Application.persistentDataPath + "/Arrays/" + name + ".txt", false);
        writer.Write(serializedData);
        UnityEngine.Debug.Log("Saved to: " + name);
        writer.Close();
    }
    public static void ArrayToFile(string name, List<List<float>> toSave) {
        string serializedData = "";
        for (int a = 0; a < toSave.Count; a++) {
            for(int b = 0; b < toSave[a].Count; b++) {
                serializedData += toSave[a][b] + ",";
            }
            if (a != toSave.Count - 1)
                serializedData += "\n";
        }
        StreamWriter writer = new StreamWriter(Application.persistentDataPath + "/Arrays/" + name + ".txt", false);
        writer.Write(serializedData);
        UnityEngine.Debug.Log("Saved to: " + name);
        writer.Close();
    }
    public static float[] Normalize(float[] array) {
        float length = 0;
        float[] array_ = new float[array.Length];
        for(int a =0; a < array.Length; a++) {
            length += (array[a] * array[a]);
        }
        length = Mathf.Sqrt(length);
        for (int a = 0; a < array.Length; a++) {
            array_[a] = array[a] / length;
        }
        return array;
    }
    public static float[] Discount(float[] array, float discountRate) {
        float[] array_ = new float[array.Length];
        for (int a = array.Length - 2; a >= 0; a--) {
            array_[a] = array_[a + 1] * discountRate + array[a];
        }
        return array_;
    }
    public static float[] AddArrays(float[] array, float[] array2) {
        float[] array_ = new float[array.Length];
        for (int a = 0; a < array.Length; a++) {
            array_[a] = array[a] + array2[a];
        }
        return array_;
    }
    public static float[] GetMSEErrors(float[] prediction, float[] truth) {
        float[] error = new float[prediction.Length];
        for (int a = 0; a < prediction.Length; a++) {
            error[a] = prediction[a] - truth[a];
        }
        return error;
    }
    public static void SlowMath() {
        double slowMath = Math.Exp(1.23456789) * Math.Exp(1.23456789) * Math.Exp(1.23456789);
    }
}

