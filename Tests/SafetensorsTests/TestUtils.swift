import Foundation

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
