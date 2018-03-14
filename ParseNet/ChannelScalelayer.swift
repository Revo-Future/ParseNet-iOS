//
//  ChannelScale.swift
//  ParseNet
//
//  Created by Wenbo Huang on 2017/8/7.
//  Copyright © 2017年 Hollance. All rights reserved.
//

import Foundation


import Metal
import MetalPerformanceShaders
import Accelerate


/**
 maxout layer
 */
public class ChannelScale {
    let pipeline: MTLComputePipelineState
    let weightsBuffer: MTLBuffer
    
    public let device: MTLDevice
    
    public var offset = MPSOffset(x: 0, y: 0, z: 0)
    public var clipRect = MPSRectNoClip
    public var destinationFeatureChannelOffset = 0
    public var edgeMode = MPSImageEdgeMode.zero
    /**
     Creates a new maxout object.
     
     - Parameters:
     - featureChannels: the feature channel of input feature map.
     - relu: If true, applies a ReLU to the output. Default is false.
     */
    public init(device: MTLDevice,
                featureChannels: Int,
                scaleWeights: UnsafePointer<Float>,
                width: Int = 32,
                height: Int = 32 ) {
        
        let inputSlices = (featureChannels + 3) / 4
        let paddedInputChannels = inputSlices * 4
        
        weightsBuffer = device.makeBuffer(length: MemoryLayout<UInt16>.stride * paddedInputChannels)
        let ptr = UnsafeMutablePointer(mutating: scaleWeights)
        float32to16(input: ptr, output: weightsBuffer.contents(), count: paddedInputChannels)
        
        
        let functionName: String
        if featureChannels <= 4 {
            functionName = "channelscale"
        } else {
            functionName = "channelscale_arry"
        }
        
        pipeline = makeFunction(device: device, name: functionName, useForgeLibrary: false)
        
        self.device = device
    }
    
    public func encode(commandBuffer: MTLCommandBuffer,
                       sourceImage: MPSImage, destinationImage: MPSImage) {
        
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(sourceImage.texture, at: 0)
        encoder.setTexture(destinationImage.texture, at: 1)
        encoder.setBuffer(weightsBuffer, offset: 0, at: 0)
        encoder.dispatch(pipeline: pipeline, image: destinationImage)
        encoder.endEncoding()
        
        if let image = sourceImage as? MPSTemporaryImage {
            image.readCount -= 1
        }
    }
}
