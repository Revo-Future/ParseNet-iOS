#include <metal_stdlib>

using namespace metal;

/*
 The input texture has four 16-bit floats per pixel, in the range 0.0...1.0.
 This shader function converts those floats to the range -128...127.
 
 The values we subtract from the R/G/B components are the mean R/G/B values
 across the set of images that the neural network was trained on.
 
 The alpha component of outColor is not important, since our MPSImages only
 use the first 3 feature channels.
 
 NOTE: We flip RGB textures to BGR (inColor.x and inColor.z get swapped),
 since the tool that was used to train the network, Caffe, uses images with
 BGR pixels. Therefore outColor.x is always B and outColor.y is always R.
 */

kernel void adjust_mean_rgb(texture2d<half, access::read> inTexture [[texture(0)]],
                            texture2d<half, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
    //    half4 inColor = inTexture.read(gid);
    //    half4 outColor = half4(inColor.z*255.0h - 103.939h, inColor.y*255.0h - 116.779h, inColor.x*255.0h - 123.68h, 0.0h);
    //    outTexture.write(outColor, gid);
    
    
    const auto means = float4(123.68f, 116.78f, 103.94f, 0.0f);
    const auto inColor = (float4(inTexture.read(gid)) * 255.0f - means) * 0.017f;
    outTexture.write(half4(inColor.z, inColor.y, inColor.x, 0.0f), gid);
}

kernel void adjust_mean_bgr(texture2d<half, access::read> inTexture [[texture(0)]],
                            texture2d<half, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
    //    half4 inColor = inTexture.read(gid);
    //    half4 outColor = half4(inColor.x*255.0h - 103.939h, inColor.y*255.0h - 116.779h, inColor.z*255.0h - 123.68h, 0.0h);
    //    outTexture.write(outColor, gid);
    
    
    const auto means = float4(123.68f, 116.78f, 103.94f, 0.0f);
    const auto inColor = (float4(inTexture.read(gid)) * 255.0f - means) * 0.017f;
    outTexture.write(half4(inColor.x, inColor.y, inColor.z, 0.0f), gid);
}


enum NeuronType: ushort {
    NeuronTypeNone = 0,
    NeuronTypeReLU = 1,
    NeuronTypeLinear = 2,
    NeuronTypeSigmoid = 3,
    NeuronTypeTanH = 4,
    NeuronTypeAbsolute = 5,
    };
    
    constant ushort kernelWidth [[ function_constant(0) ]];
    constant ushort kernelHeight [[ function_constant(1) ]];
    constant ushort2 stride [[ function_constant(2) ]];
    constant ushort neuronType [[ function_constant(3) ]];
    
    struct KernelParams {
        ushort inputWidth;
        ushort inputHeight;
        ushort inputFeatureChannels;
        ushort inputSlices;
        ushort inputOffsetX;
        ushort inputOffsetY;
        ushort inputOffsetZ;
        ushort outputWidth;
        ushort outputHeight;
        ushort outputFeatureChannels;
        ushort outputSlices;
        ushort destinationSliceOffset;
        ushort outputOffsetX;
        ushort outputOffsetY;
        ushort outputOffsetZ;
        ushort edgeMode;
        float neuronA;
        float neuronB;
    };
    
    // Applying the activation function in the shader is quicker than creating
    // a new layer for it.
    inline float4 applyNeuron(float4 x, float a, float b) {
        if (neuronType == NeuronTypeReLU)
            return fmax(x, 0.0f) + a*fmin(x, 0.0f);
        if (neuronType == NeuronTypeLinear)
            return a*x + b;
        if (neuronType == NeuronTypeSigmoid)
            return 1.0f / (1.0f + exp(-x));
        if (neuronType == NeuronTypeTanH)
            return a * tanh(b * x);
        if (neuronType == NeuronTypeAbsolute)
            return fabs(x);
        return x;
    }
    
    inline half4 applyNeuron(half4 x, half a, half b) {
        if (neuronType == NeuronTypeReLU)
            return fmax(x, 0.0h) + a*fmin(x, 0.0h);
        if (neuronType == NeuronTypeLinear)
            return a*x + b;
        if (neuronType == NeuronTypeSigmoid)
            return 1.0h / (1.0h + exp(-x));
        if (neuronType == NeuronTypeTanH)
            return a * tanh(b * x);
        if (neuronType == NeuronTypeAbsolute)
            return fabs(x);
        return x;
    }
    
    // MARK: - Preprocessing kernels
    
    kernel void rgb2Gray(
                         texture2d<half, access::read> inTexture [[texture(0)]],
                         texture2d<half, access::write> outTexture [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height()) {
            return;
        }
        const half4 inColor = inTexture.read(gid);
        const half y = inColor.x*0.299h + inColor.y*0.587h + inColor.z*0.114h;
        outTexture.write(half4(y * 255.0h, 0.0h, 0.0h, 0.0h), gid);
    }
    
    kernel void rgb2bgr(
                        texture2d<half, access::read> inTexture [[texture(0)]],
                        texture2d<half, access::write> outTexture [[texture(1)]],
                        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height()) {
            return;
        }
        const half4 inColor = inTexture.read(gid);
        outTexture.write(half4(inColor.z, inColor.y, inColor.x, 0.0h), gid);
    }
    
    kernel void subtractMeanColor(
                                  texture2d<half, access::read> inTexture [[texture(0)]],
                                  texture2d<half, access::write> outTexture [[texture(1)]],
                                  constant half4& params [[buffer(0)]],
                                  uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height()) {
            return;
        }
        const half4 meanColor = params[0];
        const half4 meanScale = params[1];
        outTexture.write(inTexture.read(gid) * meanScale - meanColor, gid);
    }
    
    //added by Guigu
    kernel void eltwiseSum(
                              texture2d<half, access::read> inTexture1 [[texture(0)]],
                              texture2d<half, access::read> inTexture2 [[texture(1)]],
                              texture2d<half, access::write> outTexture [[texture(2)]],
                              //constant KernelParams& params [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height()) {
            return;
        }
        half4 out = half4(0.0f);
//
        out = inTexture1.read(gid)+inTexture2.read(gid);

        outTexture.write(out, gid);
        
    }
    kernel void eltwiseSum_array(
                                  texture2d_array<half, access::sample> inTexture1 [[texture(0)]],
                                  texture2d_array<half, access::sample> inTexture2 [[texture(1)]],
                                  texture2d_array<half, access::write> outTexture [[texture(2)]],
                                  ushort3 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort2 pos = gid.xy;
        const ushort slices = outTexture.get_array_size();
        const ushort slice = gid.z;
        
        half4 in[2];
        in[0] = inTexture1.sample(s, float2(pos.x, pos.y), slice);
        in[1] = inTexture2.sample(s, float2(pos.x, pos.y), slice);
        
        float4 out = float4(0.0f);
        
        out =float4( in[0]+in[1]);
        
        //out = applyNeuron(out, params.neuronA, params.neuronB);
        
        outTexture.write(half4(out), gid.xy, gid.z);

    }
//    kernel void eltwise_array(
//                              texture2d_array<half, access::read> inTexture1 [[texture(0)]],
//                              texture2d_array<half, access::read> inTexture2 [[texture(1)]],
//                              texture2d_array<half, access::write> outTexture [[texture(2)]],
//                              constant KernelParams& params [[buffer(0)]],
//                              uint2 gid [[thread_position_in_grid]])
//    {
//        if (gid.x >= outTexture.get_width() ||
//            gid.y >= outTexture.get_height()
//           ) return;
//        //
//        ////        const ushort2 pos = gid.xy * stride + ushort2(params.inputOffsetX, params.inputOffsetY);
//        //        const ushort inSlices = inTexture1.get_array_size();
//        //        const ushort outSlice = gid.z;
//        
//        float4 out = float4(0.0f);
//        
//        out = half4(inTexture1.read(gid)+inTexture2.read(gid));
//        
//        out = applyNeuron(out, params.neuronA, params.neuronB);
//        
//        outTexture.write(half4(out), gid);
//    }
    
    // MARK: - Convolution
    
    /*
     Very basic implementation of convolution. Don't use this in production code;
     it's just for testing Forge and running experiments.
     */
    
    kernel void conv3x3(
                        texture2d<half, access::sample> inTexture [[texture(0)]],
                        texture2d<half, access::write> outTexture [[texture(1)]],
                        constant KernelParams& params [[buffer(0)]],
                        const device half4* weights [[buffer(1)]],
                        const device half4* biasTerms [[buffer(2)]],
                        ushort3 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort kW = 3;
        const ushort kH = 3;
        
        const ushort2 pos = gid.xy * stride + ushort2(params.inputOffsetX, params.inputOffsetY);
        
        // Note: If we use half4, then we lose too much precision.
        float4 out = float4(0.0f);
        
        half4 in[9];
        in[0] = inTexture.sample(s, float2(pos.x - 1, pos.y - 1));
        in[1] = inTexture.sample(s, float2(pos.x    , pos.y - 1));
        in[2] = inTexture.sample(s, float2(pos.x + 1, pos.y - 1));
        in[3] = inTexture.sample(s, float2(pos.x - 1, pos.y    ));
        in[4] = inTexture.sample(s, float2(pos.x    , pos.y    ));
        in[5] = inTexture.sample(s, float2(pos.x + 1, pos.y    ));
        in[6] = inTexture.sample(s, float2(pos.x - 1, pos.y + 1));
        in[7] = inTexture.sample(s, float2(pos.x    , pos.y + 1));
        in[8] = inTexture.sample(s, float2(pos.x + 1, pos.y + 1));
        
        for (ushort t = 0; t < kH*kW; ++t) {
            half4 wx = weights[0*kH*kW + t];
            out.x += dot(float4(in[t]), float4(wx));
            
            half4 wy = weights[1*kH*kW + t];
            out.y += dot(float4(in[t]), float4(wy));
            
            half4 wz = weights[2*kH*kW + t];
            out.z += dot(float4(in[t]), float4(wz));
            
            half4 ww = weights[3*kH*kW + t];
            out.w += dot(float4(in[t]), float4(ww));
        }
        
        out += float4(biasTerms[0]);
        out = applyNeuron(out, params.neuronA, params.neuronB);
        
        outTexture.write(half4(out), gid.xy);
    }
    
    kernel void conv3x3_array(
                              texture2d_array<half, access::sample> inTexture [[texture(0)]],
                              texture2d_array<half, access::write> outTexture [[texture(1)]],
                              constant KernelParams& params [[buffer(0)]],
                              const device half4* weights [[buffer(1)]],
                              const device half4* biasTerms [[buffer(2)]],
                              ushort3 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort kW = 3;
        const ushort kH = 3;
        
        const ushort2 pos = gid.xy * stride + ushort2(params.inputOffsetX, params.inputOffsetY);
        const ushort inSlices = inTexture.get_array_size();
        const ushort outSlice = gid.z;
        
        float4 out = float4(0.0f);
        
        for (ushort inSlice = 0; inSlice < inSlices; ++inSlice) {
            half4 in[9];
            in[0] = inTexture.sample(s, float2(pos.x - 1, pos.y - 1), inSlice);
            in[1] = inTexture.sample(s, float2(pos.x    , pos.y - 1), inSlice);
            in[2] = inTexture.sample(s, float2(pos.x + 1, pos.y - 1), inSlice);
            in[3] = inTexture.sample(s, float2(pos.x - 1, pos.y    ), inSlice);
            in[4] = inTexture.sample(s, float2(pos.x    , pos.y    ), inSlice);
            in[5] = inTexture.sample(s, float2(pos.x + 1, pos.y    ), inSlice);
            in[6] = inTexture.sample(s, float2(pos.x - 1, pos.y + 1), inSlice);
            in[7] = inTexture.sample(s, float2(pos.x    , pos.y + 1), inSlice);
            in[8] = inTexture.sample(s, float2(pos.x + 1, pos.y + 1), inSlice);
            
            for (ushort t = 0; t < kH*kW; ++t) {
                half4 wx = weights[(outSlice*4 + 0)*kH*kW*inSlices + t*inSlices + inSlice];
                out.x += dot(float4(in[t]), float4(wx));
                
                half4 wy = weights[(outSlice*4 + 1)*kH*kW*inSlices + t*inSlices + inSlice];
                out.y += dot(float4(in[t]), float4(wy));
                
                half4 wz = weights[(outSlice*4 + 2)*kH*kW*inSlices + t*inSlices + inSlice];
                out.z += dot(float4(in[t]), float4(wz));
                
                half4 ww = weights[(outSlice*4 + 3)*kH*kW*inSlices + t*inSlices + inSlice];
                out.w += dot(float4(in[t]), float4(ww));
            }
        }
        
        out += float4(biasTerms[outSlice]);
        out = applyNeuron(out, params.neuronA, params.neuronB);
        
        outTexture.write(half4(out), gid.xy, outSlice);
    }
    
    
    // MARK: - Depth-wise convolution
    
    kernel void depthwiseConv3x3(
                                 texture2d<half, access::sample> inTexture [[texture(0)]],
                                 texture2d<half, access::write> outTexture [[texture(1)]],
                                 constant KernelParams& params [[buffer(0)]],
                                 const device half4* weights [[buffer(1)]],
                                 const device half4* biasTerms [[buffer(2)]],
                                 ushort2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        // Note: this is a very naive implementation of convolution.
        // There are ways to make it a lot faster...
        
        // Seen from the destination image, the stride is how far apart the pixels
        // are in the source image.
        const ushort2 pos = gid * stride + ushort2(params.inputOffsetX, params.inputOffsetY);
        
        // Read the 3x3 pixels surrounding the source pixel.
        // By processing the pixels as half4 values we do up to 4 channels at a time.
        half4 in[9];
        in[0] = inTexture.sample(s, float2(pos.x - 1, pos.y - 1));
        in[1] = inTexture.sample(s, float2(pos.x    , pos.y - 1));
        in[2] = inTexture.sample(s, float2(pos.x + 1, pos.y - 1));
        in[3] = inTexture.sample(s, float2(pos.x - 1, pos.y    ));
        in[4] = inTexture.sample(s, float2(pos.x    , pos.y    ));
        in[5] = inTexture.sample(s, float2(pos.x + 1, pos.y    ));
        in[6] = inTexture.sample(s, float2(pos.x - 1, pos.y + 1));
        in[7] = inTexture.sample(s, float2(pos.x    , pos.y + 1));
        in[8] = inTexture.sample(s, float2(pos.x + 1, pos.y + 1));
        
        // Multiply by the weights and put the weighted sum in the output pixel.
        // Do these calculations as 32-bit float or we lose too much precision.
        float4 out = float4(0.0f);
        for (ushort t = 0; t < 9; ++t) {
            out += float4(in[t]) * float4(weights[t]);
        }
        
        out += float4(biasTerms[0]);
        
        out = applyNeuron(out, params.neuronA, params.neuronB);
        
        outTexture.write(half4(out), gid);
    }
    
    kernel void depthwiseConv3x3_array(
                                       texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                       texture2d_array<half, access::write> outTexture [[texture(1)]],
                                       constant KernelParams& params [[buffer(0)]],
                                       const device half4* weights [[buffer(1)]],
                                       const device half4* biasTerms [[buffer(2)]],
                                       ushort3 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort2 pos = gid.xy * stride + ushort2(params.inputOffsetX, params.inputOffsetY);
        const ushort slices = outTexture.get_array_size();
        const ushort slice = gid.z;
        
        half4 in[9];
        in[0] = inTexture.sample(s, float2(pos.x - 1, pos.y - 1), slice);
        in[1] = inTexture.sample(s, float2(pos.x    , pos.y - 1), slice);
        in[2] = inTexture.sample(s, float2(pos.x + 1, pos.y - 1), slice);
        in[3] = inTexture.sample(s, float2(pos.x - 1, pos.y    ), slice);
        in[4] = inTexture.sample(s, float2(pos.x    , pos.y    ), slice);
        in[5] = inTexture.sample(s, float2(pos.x + 1, pos.y    ), slice);
        in[6] = inTexture.sample(s, float2(pos.x - 1, pos.y + 1), slice);
        in[7] = inTexture.sample(s, float2(pos.x    , pos.y + 1), slice);
        in[8] = inTexture.sample(s, float2(pos.x + 1, pos.y + 1), slice);
        
        float4 out = float4(0.0f);
        for (ushort t = 0; t < 9; ++t) {
            out += float4(in[t]) * float4(weights[t*slices + slice]);
        }
        
        out += float4(biasTerms[slice]);
        
        out = applyNeuron(out, params.neuronA, params.neuronB);
        
        outTexture.write(half4(out), gid.xy, gid.z);
    }
    
    // MARK: - Transpose channels
    
    kernel void transposeChannels(
                                  texture2d<half, access::read> inTexture [[texture(0)]],
                                  texture2d<half, access::write> outTexture [[texture(1)]],
                                  const device ushort* permute [[buffer(0)]],
                                  ushort2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height()) return;
        
        const half4 in = inTexture.read(gid);
        const half4 out = half4(in[permute[0]], in[permute[1]], in[permute[2]], in[permute[3]]);
        outTexture.write(out, gid);
    }
    
    kernel void transposeChannels_array(
                                        texture2d_array<half, access::read> inTexture [[texture(0)]],
                                        texture2d_array<half, access::write> outTexture [[texture(1)]],
                                        const device ushort* permute [[buffer(0)]],
                                        ushort3 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        
        half4 out = half4(0.0h);
        
        for (ushort i = 0; i < 4; ++i) {
            const ushort perm = permute[(gid.z << 2) + i];
            const ushort slice = perm >> 2;
            const ushort comp = perm - (slice << 2);
            const half4 in = inTexture.read(gid.xy, slice);
            out[i] = in[comp];
        }
        
        outTexture.write(out, gid.xy, gid.z);
    }
    
    kernel void unpooling_array(
                                       texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                       texture2d_array<half, access::write> outTexture [[texture(1)]],
                                       ushort3 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort slice = gid.z;
        
        half4 out = inTexture.sample(s, float2(0, 0), slice);
        
        outTexture.write(half4(out), gid.xy, gid.z);
    }
    
    
    
    inline uint getMaxIndex(half4 input) {
        uint index = 1;
        half tmp = input.x;
        if (input.y > tmp){
            index = 2;
            tmp = input.y;
        }
        if (input.z > tmp){
            index = 3;
            tmp = input.z;
        }
        if (input.w > tmp){
            index = 4;
        }
      
        return index;
    }
    kernel void maxout_array(
                                texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                texture2d<half, access::write> outTexture [[texture(1)]],
                                ushort2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort2 pos = gid.xy;
        const ushort inSlices = inTexture.get_array_size();
        
        half4 input=half4(0.0h);
        
        //half tmp = inTexture.sample(s, float2(0,0), 0).w;
        half tmp = -900;
        uint index = 1;
        uint total_index = 1;
        
        for (ushort inSlice = 0; inSlice < inSlices; ++inSlice) {
            input = inTexture.sample(s, float2(pos.x, pos.y), inSlice);
            if (input.x >= tmp){
                index = 1;
                tmp = input.x;
                total_index = 4*inSlice + index;
            }
            if (input.y > tmp){
                index = 2;
                tmp = input.y;
                total_index = 4*inSlice + index;
            }
            if (input.z > tmp){
                index = 3;
                tmp = input.z;
                total_index = 4*inSlice + index;
            }
            if (input.w > tmp){
                index = 4;
                tmp = input.w;
                total_index = 4*inSlice + index;
            }
            
          
        }
        
        outTexture.write(total_index-1, gid.xy);
    }
    
    kernel void channelscale_arry(
                             texture2d_array<half, access::sample> inTexture [[texture(0)]],
                             texture2d_array<half, access::write> outTexture [[texture(1)]],
                             const device half4* weights [[buffer(0)]],
                             ushort3 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort2 pos = gid.xy;
        //const ushort slices = outTexture.get_array_size();
        const ushort slice = gid.z;
        
        half4 in;
        in = inTexture.sample(s, float2(pos.x    , pos.y    ), slice);
        
        half4 scaleweight = weights[slice];
        
        float4 out = float4(0.0f);
        
        out = float4(in) * float4(scaleweight);
        
        outTexture.write(half4(out), gid.xy, gid.z);
        
        
    }
    
    kernel void resize_interpolation_array(
                                  texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                  texture2d_array<half, access::write> outTexture [[texture(1)]],
                                  ushort3 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort2 pos = gid.xy;
        const ushort slice = gid.z;
        
        float scale_x = float(inTexture.get_width())/float(outTexture.get_width());
        float scale_y = float(inTexture.get_height())/float(outTexture.get_height());
        
        float i = scale_x * float(pos.x+0.5)-0.5;
        float j = scale_y * float(pos.y+0.5)-0.5;
        float i_ceil = ceil(i);
        float i_floor = floor(i);
        float j_ceil = ceil(j);
        float j_floor = floor(j);
        
        half4 in;
        in = inTexture.sample(s, float2(pos.x    , pos.y    ), slice);
        
        half4 f00 = inTexture.sample(s,float2(i_floor,j_floor),slice);
        half4 f10 = inTexture.sample(s,float2(i_ceil,j_floor),slice);
        half4 f01 = inTexture.sample(s,float2(i_floor,j_ceil),slice);
        half4 f11 = inTexture.sample(s,float2(i_ceil,j_ceil),slice);
        
        float4 r1 =(i_ceil-i)*float4(f01) + (i-i_floor)*float4(f11);
        float4 r2 = (i_ceil-i)*float4(f00) + (i-i_floor)*float4(f10);
        float4 out = (j-j_floor)*r1 + (j_ceil -j)*r2;
        
        outTexture.write(half4(out), gid.xy, gid.z);
    }
    
    kernel void pallete(
                                    texture2d<half, access::sample> inTexture [[texture(0)]],
                                    texture2d<half, access::write> outTexture [[texture(1)]],
                                    ushort2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort2 pos = gid.xy;
        
        int4 label =int4( inTexture.sample(s,float2(pos.x,pos.y)));
        half4 out = half4(0.0f);
        switch(label.x)
        {
            case 0:
                out = half4(0,0,0,0);
                break;
            case 1:
                out = half4(128,0,0,255);
                break;
            case 2:
                out = half4(0,128,0,255);
                break;
            case 3:
                out = half4(128,128,0,0);
                break;
            case 4:
                out = half4(0,0,128,0);
                break;
            case 5:
                out = half4(128,0,128,0);
                break;
            case 6:
                out = half4(0,128,128,0);
                break;
            case 7:
                out = half4(128.0,128.0,128.0,255);
                break;
            case 8:
                out = half4(64,0,0,0);
                break;
            case 9:
                out = half4(192,0,0,0);
                break;
            case 10:
                out = half4(64,128,0,0);
                break;
            case 11:
                out = half4(192,128,0,255);
                break;
            case 12:
                out = half4(64,0,128,0);
                break;
            case 13:
                out = half4(192,0,128,0);
                break;
            case 14:
                out = half4(64,128,128,255);
                break;
            case 15:
                out = half4(192,128,128,255);
                break;
            case 16:
                out = half4(0,64,0,0);
                break;
            case 17:
                out = half4(128,64,0,0);
                break;
            default:
                out = half4(0,0,0,255);
                break;
        }
        
        
        out.w = 255.0;
        outTexture.write(half4(out), gid.xy);
    }
    
