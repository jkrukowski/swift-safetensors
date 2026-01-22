import Foundation

public enum EncodedSafetensors {
    case single(Data)
    case sharded(ShardedData)
}
