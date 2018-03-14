//
//  MultiChannelResizelayer.swift
//  ParseNet
//
//  Created by Wenbo Huang on 2017/8/13.
//  Copyright © 2017年 Hollance. All rights reserved.
//

import Foundation

import Metal
import MetalPerformanceShaders
import Accelerate


/**
multiple channel feature map resize layer
 */
public class MultiChannelResizeKernel {
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
                featureChannels: Int
                ) {
        let functionName: String
        if featureChannels <= 4 {
            functionName = "resize_interpolation"
        } else {
            functionName = "resize_interpolation_array"
        }
        
        pipeline = makeFunction(device: device, name: functionName, useForgeLibrary: false)
        self.device = device
    }
    
    public func encode(commandBuffer: MTLCommandBuffer,
                       sourceImage: MPSImage, destinationImage: MPSImage) {
        
        // TODO: set the KernelParams based on clipRect, destinationFeatureChannelOffset, edgeMode
        
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
