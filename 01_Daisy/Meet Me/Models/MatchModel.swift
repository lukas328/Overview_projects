//
//  MatchModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 27.02.21.
//

import Foundation


struct MatchModel: Codable {
    var userId: String
    var chatId : String
    var eventId : String
    var matchedUserId: String
    var unReadMessage: Bool
}
