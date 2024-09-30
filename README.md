# `swift-safetensors`

Swift package for reading and writing [Safetensors](https://github.com/huggingface/safetensors) files.

## Installation

Add the following to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/jkrukowski/swift-safetensors", from: "0.0.1")
]
```

## Usage

### Read `Safetensors` file

```swift
import Safetensors

let safetensors = try Safetensors.read(at: "path/to/file.safetensors")

// get MLTensor
let mlTensor = try safetensors.mlTensor(forKey: "tensorKey")

// get MLMultiArray
let mlMultiArray = try safetensors.mlMultiArray(forKey: "tensorKey")
```

### Write `Safetensors` file

```swift
import Safetensors

let data: [String: any SafetensorsEncodable] = [
    "test1": MLMultiArray(MLShapedArray<Int32>(repeating: 1, shape: [2, 2])),
    "test2": MLMultiArray(MLShapedArray<Int32>(repeating: 2, shape: [9])),
]

try Safetensors.write(data, to: "path/to/file.safetensors")
```
