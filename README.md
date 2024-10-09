# `swift-safetensors`

[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fjkrukowski%2Fswift-safetensors%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/jkrukowski/swift-safetensors)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fjkrukowski%2Fswift-safetensors%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/jkrukowski/swift-safetensors)

Swift package for reading and writing [Safetensors](https://github.com/huggingface/safetensors) files.

## Installation

Add the following to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/jkrukowski/swift-safetensors", from: "0.0.5")
]
```

## Usage

### Read `Safetensors` file

```swift
import Safetensors

let parsedSafetensors = try Safetensors.read(at: URL(filePath: "path/to/file.safetensors"))

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
```

When `MLTensor` or `MLMultiArray` is materialized, the data is copied from the underlying buffer.
If you want to avoid copying, you can do:

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

But make sure that the `ParsedSafetensors` object is not deallocated before you finish using the `MLTensor`, `MLMultiArray` or `MLShapedArray`.

### Write `Safetensors` file

```swift
import Safetensors

let data: [String: any SafetensorsEncodable] = [
    "test1": MLMultiArray(MLShapedArray<Int32>(repeating: 1, shape: [2, 2])),
    "test2": MLMultiArray(MLShapedArray<Int32>(repeating: 2, shape: [9])),
]

try Safetensors.write(data, to: URL(filePath: "path/to/file.safetensors"))
```

## Code Formatting

This project uses [swift-format](https://github.com/swiftlang/swift-format). To format the code run:

```bash
swift-format format . -i -r --configuration .swift-format
```
