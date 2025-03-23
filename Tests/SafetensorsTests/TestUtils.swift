import Foundation

@testable import Safetensors

func createRawSafetensors(
    headerSize: UInt64,
    headerString: String,
    tensorData: Data
) -> Data {
    let headerSizeData = withUnsafeBytes(of: headerSize) { Data($0) }
    return createRawSafetensors(
        headerSizeData: headerSizeData,
        headerString: headerString,
        tensorData: tensorData
    )
}

func createRawSafetensors(
    headerSizeData: Data,
    headerString: String,
    tensorData: Data
) -> Data {
    let headerData = headerString.data(using: .utf8)!
    return headerSizeData + headerData + tensorData
}

func createRawSafetensors(
    headerString: String,
    tensorData: Data
) -> Data {
    let headerData = headerString.data(using: .utf8)!
    let headerSizeData = withUnsafeBytes(of: UInt64(headerData.count)) { Data($0) }
    return headerSizeData + headerData + tensorData
}

func writeToTemporaryFile(
    _ fileName: String,
    data: [String: any SafetensorsEncodable]
) throws -> URL {
    let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
    try Safetensors.write(data, to: fileURL)
    return fileURL
}

final class TestTensor: SafetensorsEncodable {
    let size: Int
    let shape: [Int]

    init(size: Int, shape: [Int]) {
        self.size = size
        self.shape = shape
    }

    var tensorScalarCount: Int {
        shape.reduce(1, *)
    }

    var tensorShape: [Int] {
        shape
    }

    var dtype: String {
        Constants.DataType.float32
    }

    var scalarSize: Int {
        4  // Size of Float32
    }

    func toData() throws -> Data {
        Data(count: size)
    }

    var tensorByteCount: Int {
        size
    }
}
