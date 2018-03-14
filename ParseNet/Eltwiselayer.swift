//
//  Eltwise.swift
//  MobileNet
//
//  Created by Wenbo Huang on 17/7/5.
//  Copyright © 2017年 Hollance. All rights reserved.
//

import Metal
import MetalPerformanceShaders
import Accelerate


/**
 element-wise layer
 
 currently, only support sum operation
 */
public class EltwiseKernel {
    let pipeline: MTLComputePipelineState
//    let weightsBuffer: MTLBuffer
//    let biasBuffer: MTLBuffer
    
    
    public let device: MTLDevice
    
    public var offset = MPSOffset(x: 0, y: 0, z: 0)
    public var clipRect = MPSRectNoClip
    public var destinationFeatureChannelOffset = 0
    public var edgeMode = MPSImageEdgeMode.zero
    /**
     Creates a new elementwise sum object.
     
     - Parameters:
     - featureChannels: the feature channel of input feature map.
     - relu: If true, applies a ReLU to the output. Default is false.
     */
    public init(device: MTLDevice,
                featureChannels: Int,
                strideInPixelsX: Int = 1,
                strideInPixelsY: Int = 1) {
 
        //let inputSlices = (featureChannels + 3) / 4
        //let paddedInputChannels = inputSlices * 4

        
        let functionName: String
        if featureChannels <= 4 {
            functionName = "eltwiseSum"
        } else {
            functionName = "eltwiseSum_array"
        }
        
        
        pipeline = makeFunction(device: device, name: functionName, useForgeLibrary: false)
        self.device = device
    }
    
    public func encode(commandBuffer: MTLCommandBuffer,
                       sourceImage1: MPSImage, sourceImage2: MPSImage, destinationImage: MPSImage) {
        
        // TODO: set the KernelParams based on clipRect, destinationFeatureChannelOffset, edgeMode
        
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(sourceImage1.texture, at: 0)
        encoder.setTexture(sourceImage2.texture, at: 1)
        encoder.setTexture(destinationImage.texture, at: 2)
        //encoder.setBytes(&params, length: MemoryLayout<KernelParams>.size, at: 0)
        encoder.dispatch(pipeline: pipeline, image: destinationImage)
        encoder.endEncoding()
        
        if let image = sourceImage1 as? MPSTemporaryImage {
            image.readCount -= 1
        }
    }
}
