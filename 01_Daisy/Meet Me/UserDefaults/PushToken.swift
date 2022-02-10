//
//  PushToken.swift
//  Meet Me
//
//  Created by Lukas Dech on 12.04.21.
//

import Foundation

class PushTokens: ObservableObject {
    @Published var token: [String: String]
    static let saveKey = "PushTokens"
    
    init() {
        if let data = UserDefaults.standard.data(forKey: Self.saveKey) {
            if let decoded = try? JSONDecoder().decode([String: String].self, from: data) {
                
                self.token = decoded
                return
            }
        }
        self.token = [String:String]()
    }
    
    func save() {
        if let encoded = try? JSONEncoder().encode(token) {
            UserDefaults.standard.set(encoded, forKey: Self.saveKey)
        }
    }

    
}
