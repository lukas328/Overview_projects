//
//  MeMatchStartViewModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 21.02.21.
//

import Foundation
import PromiseKit

class MeEventControllerViewModel: ObservableObject {
    
    private var firestoreManagerEventTest: FirestoreManagerEventTest = FirestoreManagerEventTest()
    @Published var likedUsers: [UserModel] = []
    
    func getLikedUsers(eventId: String) {
        
            firstly{
                self.firestoreManagerEventTest.getAllLikedUserDocument(eventId: eventId)
            }.then { likedUser in
                self.firestoreManagerEventTest.getAllLikedUserModels(likedUser: likedUser)
            }.done { users in
                self.likedUsers = users
            }.catch { error in
                print("DEBUG: error bei gettting likedUsers, error: \(error)")
                print("DEBUG: error localized \(error.localizedDescription)")
            }
        }
    }
    


