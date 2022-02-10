//
//  ChatModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 27.02.21.
//

import Foundation
import Firebase

struct ChatModel: Codable {
    var chatId: String
    var eventCreatorId: String
    var matchedUserId: String
    var eventId: String
    var messages: [MessageModel]

}

struct MessageModel: Codable {
    var userId: String
    var timeStamp: Timestamp
    var messageText: String
    
    var dictionary: [String: Any] {
        return["userId": userId,
               "timeStamp": timeStamp,
               "messageText": messageText]
    }
}
