//
//  ChangeMeEventViewModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 10.03.21.
//

import Foundation
import PromiseKit
import SwiftUI


class ChangeMeEventViewModel: ObservableObject {
    
    @Published var event: EventModel = stockEvent
    private var firestoreManagerMatches: FirestoreManagerMatches = FirestoreManagerMatches()
    private var firestoreEventTest: FirestoreManagerEventTest = FirestoreManagerEventTest()
    private var firestoreFotoManagerEventTest: FirestoreFotoManagerEventTest = FirestoreFotoManagerEventTest()
    private var chatModel: ChatModel = stockChat
    
    
    //Wenn EventPicture nicht geändert wurde nil übergeben
    func changeMeEventSettings(eventModel: EventModel, eventPicture: UIImage?) {
        firstly{
            firestoreEventTest.setUpdatedEvent(eventModel: eventModel)
        }.catch { error in
            print("DEBUG: Error in changeMeEventSettings, error: \(error) ")
            print("DEBUG: error localized: \(error.localizedDescription)")
        }
        
        if eventPicture != nil {
            firstly{
                self.firestoreFotoManagerEventTest.resizeImage(originalImage: eventPicture)
            }.then { picture in
                self.firestoreFotoManagerEventTest.uploadEventPhoto(data: picture)
            }.then { url in
                self.firestoreFotoManagerEventTest.saveEventPhotoUrlToFirestore(url: url, eventModel: eventModel)
            }.catch { error in
                print("DEBUG: catch, Fehler in EventCreationChain\(error)")
                print(error.localizedDescription)
            }
            
        }
    }
        
    func deleteMeEvent(eventModel: EventModel) {

        firstly {
            when(fulfilled:self.firestoreManagerMatches.deleteAllLikedUserFromEvent(eventId: eventModel.eventId),
                 self.firestoreFotoManagerEventTest.deleteImageFromStorage(storageId: eventModel.eventPhotosId),
                 self.firestoreManagerMatches.deleteEvent(eventId: eventModel.eventId))
                           
        }.catch { error in
            print("DEBUG: error in deleteMeEvent, error \(error)")
            print("DEBUG: error localized: \(error.localizedDescription)")
        }
        
        if eventModel.eventMatched {
            firstly{
                firestoreManagerMatches.getChatWithEventId(eventId: eventModel.eventId)
            }.map { chat in
                self.chatModel = chat
            }.then {
                when(fulfilled:
                     self.firestoreManagerMatches.deleteMatchFromCurrentUser(chatId: self.chatModel.chatId),
                     self.firestoreManagerMatches.deleteMatchFromMatchedUser(chatId: self.chatModel.chatId, matchedUserId: self.chatModel.matchedUserId),
                     self.firestoreManagerMatches.deleteChat(chatId: self.chatModel.chatId))
            }.catch { error in
                print("DEBUG: error in deleteMeEvent, error \(error)")
                print("DEBUG: error localized: \(error.localizedDescription)")
            }
            
            
        }
        
        
    }
}
    


    
    
