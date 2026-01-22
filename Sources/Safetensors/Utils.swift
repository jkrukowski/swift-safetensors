enum Constants {
    static let metadataKey = "__metadata__"
    static let F64 = "F64"
    static let F32 = "F32"
    static let F16 = "F16"
    static let I64 = "I64"
    static let U64 = "U64"
    static let I32 = "I32"
    static let U32 = "U32"
    static let I16 = "I16"
    static let U16 = "U16"
    static let I8 = "I8"
    static let U8 = "U8"
    static let BOOL = "BOOL"
}

extension String {
    func zfill(_ width: Int) -> String {
        if self.count >= width {
            return self
        }
        let zerosNeeded = width - self.count
        if self.hasPrefix("-") {
            return "-" + String(repeating: "0", count: zerosNeeded) + self.dropFirst()
        } else {
            return String(repeating: "0", count: zerosNeeded) + self
        }
    }
}
