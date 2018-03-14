//
//  Maxoutlayer.swift
//  ParseNet
//
//  Created by Wenbo Huang on 2017/8/4.
//  Copyright © 2017年 Hollance. All rights reserved.
//

import Foundation

import Metal
import MetalPerformanceShaders
import Accelerate


/**
 maxout layer
 */
public class Maxout {
    let pipeline: MTLComputePipelineState
    
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
                width: Int = 32,
                height: Int = 32 ) {
        
        let functionName: String
        if featureChannels <= 4 {
            functionName = "maxout"
        } else {
            functionName = "maxout_array"
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
        encoder.dispatch(pipeline: pipeline, image: destinationImage)
        encoder.endEncoding()
    
        if let image = sourceImage as? MPSTemporaryImage {
            image.readCount -= 1
        }
    }
}
