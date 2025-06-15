#pragma once

class InferenceEngine 
{
    InferenceEngine();
    ~InferenceEngine();

    void loadModel(const char* path);

    void inference(const float* input, const float* output, int n);

}