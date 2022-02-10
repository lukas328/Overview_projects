//
//  MeMatchCardView.swift
//  Meet Me
//
//  Created by Lukas Dech on 23.02.21.
//

import Foundation
import PromiseKit

class MeMatchCardViewModel: ObservableObject {
    private var firestoreManagerMatches: FirestoreManagerMatches = FirestoreManagerMatches()
    private var firestoreManagerEventTest: FirestoreManagerEventTest = FirestoreManagerEventTest()
    private var firestoreManagerUserTest: FirestoreManagerUserTest = FirestoreManagerUserTest()

    
    func addMatch(eventModel: EventModel, userModel: UserModel) {
        let chatId = UUID().uuidString
            firstly {
                when(fulfilled:
                self.firestoreManagerMatches.addMatchToCurrentUser(userModel: userModel, eventModel: eventModel, chatId: chatId),
                self.firestoreManagerMatches.addMatchToMatchedUser(userModel: userModel, eventModel: eventModel, chatId: chatId),
                self.firestoreManagerEventTest.setEventMatchedToTrue(eventId: eventModel.eventId))
            }.then {
                self.firestoreManagerUserTest.getCurrentUser()
            }.then { currentUser in
                self.firestoreManagerMatches.createChatRoom(currentUser: currentUser, userModel: userModel, eventModel: eventModel, chatId: chatId)
            }.catch { error in
                print("DEBUG: Fehler in addMatchChain Error: \(error)")
                print("DEBUG: Error in addMatchChain localized: \(error.localizedDescription)")
            }
    }

        
    
    func deleteLikedUser(eventModel : EventModel, userModel: UserModel){
        print("start delete liked user")
            firstly {
                firestoreManagerMatches.deleteLikedUser(eventModel: eventModel, userModel: userModel)
            }.done {
                print("deleted user done")
            }.catch { error in
                print("DEBUG: Fehler in deleteLikedUser Error: \(error)")
                print("DEBUG: Error in deleteLikedUser localized: \(error.localizedDescription)")
            }
        }
    
    func setLikedUserToFalse(eventId: String) {
        firstly {
            firestoreManagerMatches.setLikedUserAndMatchedUserToFalse(eventId: eventId)
        }.catch { error in
            print("DEBUG: Fehler in deleteLikedUser Error: \(error)")
            print("DEBUG: Error in deleteLikedUser localized: \(error.localizedDescription)")
        }
    }
}
    

