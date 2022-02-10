//
//  likedUser.swift
//  Meet Me
//
//  Created by Lukas Dech on 26.02.21.
//

import Foundation

struct LikedUser: Codable {
    var userId: String
    var likedUser: [String]
}
