//
//  firstActions.swift
//  Meet Me
//
//  Created by Lukas Dech on 27.03.21.
//

import Foundation

class FirstActions: ObservableObject {
    @Published var firstViews: [String: Bool]
    static let saveKey = "SavedViews"
    
    init() {
        if let data = UserDefaults.standard.data(forKey: Self.saveKey) {
            if let decoded = try? JSONDecoder().decode([String: Bool].self, from: data) {
                
                self.firstViews = decoded
                return
            }
        }
        self.firstViews = [String:Bool]()
    }
    
    func save() {
        if let encoded = try? JSONEncoder().encode(firstViews) {
            UserDefaults.standard.set(encoded, forKey: Self.saveKey)
        }
    }

    
}
