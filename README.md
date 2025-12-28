# `swift-safetensors`

[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fjkrukowski%2Fswift-safetensors%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/jkrukowski/swift-safetensors)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fjkrukowski%2Fswift-safetensors%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/jkrukowski/swift-safetensors)

Swift package for reading and writing [Safetensors](https://github.com/huggingface/safetensors) files.

## Installation

Add the following to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/jkrukowski/swift-safetensors", from: "0.0.8")
]
```

## Usage

### Read `Safetensors` file

```swift
import Safetensors

let parsedSafetensors = try Safetensors.read(
    at: URL(filePath: "path/to/file.safetensors")
)

// get MLTensor
let mlTensor = try parsedSafetensors.mlTensor(
    forKey: "tensorKey"
)

// get MLMultiArray
let mlMultiArray = try parsedSafetensors.mlMultiArray(
    forKey: "tensorKey"
)

// get MLShapedArray
let mlShapedArray: MLShapedArray<Int32> = try parsedSafetensors.mlShapedArray(
    forKey: "tensorKey"
)

// get Swift Array (any numeric type)
let floatArray: [Float] = try parsedSafetensors.array(forKey: "tensorKey")
let int32Array: [Int32] = try parsedSafetensors.array(forKey: "tensorKey")
let doubleArray: [Double] = try parsedSafetensors.array(forKey: "tensorKey")

// get InlineArray (Swift 6.2+, requires macOS 26.0+)
let inlineArray: InlineArray<4, Float> = try parsedSafetensors.inlineArray(forKey: "tensorKey")
```

When `MLTensor`, `MLMultiArray` or `MLShapedArray` is materialized, the data is copied from the underlying buffer by default. If you want to avoid copying, you can use the `noCopy` parameter:

**Note:** Swift `Array` and `InlineArray` always own their data, so zero-copy access is not available for these types.

```swift
// get MLTensor without copying data
let mlTensor = try parsedSafetensors.mlTensor(
    forKey: "tensorKey",
    noCopy: true
)

// get MLMultiArray without copying data
let mlMultiArray = try parsedSafetensors.mlMultiArray(
    forKey: "tensorKey",
    noCopy: true
)

// get MLShapedArray without copying data
let mlShapedArray: MLShapedArray<Int32> = try parsedSafetensors.mlShapedArray(
    forKey: "tensorKey",
    noCopy: true
)
```

But make sure that the `ParsedSafetensors` object is not deallocated before you
finish using the `MLTensor`, `MLMultiArray` or `MLShapedArray`.

### Write `Safetensors` file

```swift
import Safetensors

let data: [String: any SafetensorsEncodable] = [
    "test1": MLShapedArray<Int32>(repeating: 1, shape: [2, 2]),
    "test2": MLShapedArray<Float32>(repeating: 2, shape: [9]),
]

try Safetensors.write(
    data,
    metadata: ["key1": "value1", "key2": "value2"],
    to: URL(filePath: "path/to/file.safetensors")
)
```

## Code Formatting

This project uses [swift-format](https://github.com/swiftlang/swift-format). To format the code run:

```bash
swift-format format . -i -r --configuration .swift-format
```
