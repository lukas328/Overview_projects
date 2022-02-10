//
//  ShowedEventsModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 24.03.21.
//

import Foundation

class ShowedEventsModel: ObservableObject {
    @Published var events: [EventModel]
    static let saveKey = "SavedData"
    
    init() {
        if let data = UserDefaults.standard.data(forKey: Self.saveKey) {
            if let decoded = try? JSONDecoder().decode([EventModel].self, from: data) {
                
                self.events = decoded
                return
            }
        }
        self.events = []
    }
    
    func save() {
        if let encoded = try? JSONEncoder().encode(events) {
            UserDefaults.standard.set(encoded, forKey: Self.saveKey)
        }
    }

    
}
