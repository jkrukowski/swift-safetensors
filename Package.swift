// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-safetensors",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .tvOS(.v16),
        .watchOS(.v9),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "Safetensors",
            targets: ["Safetensors"]
        )
    ],
    targets: [
        .target(
            name: "Safetensors"
        ),
        .testTarget(
            name: "SafetensorsTests",
            dependencies: ["Safetensors"],
            resources: [
                .copy("data.safetensors")
            ]
        ),
    ]
)
