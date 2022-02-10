//
//  InformationCardModel.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 14.03.21.
//

import Foundation
import SwiftUI

struct InformationCardModel: Codable {
    
    // heading of the card
    var headerText: String = "Do you know that?"
    // highlight (via color) heading or not
    var highlight: Bool = false
    // subheading of the card
    var footerText: String = ""
    // image of the card
    var image: String = "Chat-Boring"
    // sfSymbol in the right corner
    var sfSymbol: String = "bubble.left.and.bubble.right.fill"
    // text on the main button (forward button)
    var buttonText: String = "YES"
    // text to describe something
    var subtext: String = ""
    
}
