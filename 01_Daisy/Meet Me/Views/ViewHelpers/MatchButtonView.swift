//
//  MatchButtonLabel.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 13.03.21.
//

import SwiftUI

struct MatchButtonLabel: View {
    
    var sfSymbol: String = "xmark.circle.fill"
    var color: Color = Color.red
    
    var body: some View {
        ZStack {
            
            // MARK: Background of the button
            ZStack {
                BlurView(style: .systemUltraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .stroke(
                                LinearGradient(
                                    gradient: Gradient(stops: [
                                                        .init(color: Color(#colorLiteral(red: 0.7791666388511658, green: 0.7791666388511658, blue: 0.7791666388511658, alpha: 0.949999988079071)), location: 0),
                                                        .init(color: Color(#colorLiteral(red: 0.7250000238418579, green: 0.7250000238418579, blue: 0.7250000238418579, alpha: 0)), location: 1)]),
                                    startPoint: UnitPoint(x: 0.9016393067273221, y: 0.10416647788375455),
                                    endPoint: UnitPoint(x: 0.035519096038869824, y: 0.85416653880629)),
                                lineWidth: 0.5
                            )
                    )
                    
                color.opacity(0.1)
                    
            }
            .frame(width: 152, height: 47)
            .clipShape(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
            )
            .shadow(color: color.opacity(0.1), radius: 4, x: 0, y: 5)
            
            Image(systemName: sfSymbol)
                .font(.system(size: 34))
                .foregroundColor(color)
        }
    }
}

struct MatchButtonLabel_Previews: PreviewProvider {
    static var previews: some View {
        MatchButtonLabel()
    }
}
