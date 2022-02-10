//
//  MessageViewModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 28.02.21.
//

//Load Chat 
import Foundation
import PromiseKit
import Firebase

class MessagesViewModel: ObservableObject {
    private var firestoreManagerChat: FirestoreManagerChat = FirestoreManagerChat()
    private var firestoreMangaerUser: FirestoreManagerUserTest = FirestoreManagerUserTest()
    @Published var chat: ChatModel = stockChat
    var userId: String = Auth.auth().currentUser!.uid
    private var db: Firestore
    
    init() {
        db = Firestore.firestore()
    }
    
    func downloadChat(chatId: String) {
        firstly {
            self.downloadChatModel(chatId: chatId)
        }.catch { error in
            print("DEBUG: error in MessageDownloadChain error: \(error)")
            print("DEGUB: error localized: \(error.localizedDescription)")
        }
    }
    
    func downloadChatModel(chatId: String) -> Promise<Void> {
        return Promise { seal in
            db.collection("chats")
                .document(chatId)
                .addSnapshotListener{ (snapshot, error) in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        if let snapshot = snapshot {
                            let chatModel = try? snapshot.data(as: ChatModel.self)
                            if chatModel != nil {
                                if chatModel?.messages.count != 0 {
                                    DispatchQueue.main.async {
                                        self.chat = chatModel!
                                        seal.fulfill(())
                                    }
                                } else {
                                    self.chat = stockChat
                                    seal.fulfill(())
                                }
                            }
                        }
                    }
                }
        }
    }
    
    func UploadChat(allMatchInformation: AllMatchInformationModel, messageText: String) {
        print("upload chat")
        firstly {
            self.firestoreManagerChat.uploadMessage(messageText: messageText, chatId: allMatchInformation.chatId)
        }.then {
            self.firestoreManagerChat.setUnReadMessageToTrue(chatId: allMatchInformation.chatId, userIdReciever: allMatchInformation.user.userId)
        }.catch { error in
            print("DEBUG: error in MessageUploadChain error: \(error)")
            print("DEGUB: error localized: \(error.localizedDescription)")
            
        }
    }
    
    func unReadMessageFalse(chatId: String) {
        firstly {
            firestoreManagerChat.setUnReadMessageToFalse(chatId: chatId)
        }.catch { error in
            print(error)
        }
    }
}
